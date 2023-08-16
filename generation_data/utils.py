import datetime
import os
import re
import time
import gc

import numpy as np
import tensorflow as tf
import keras_nlp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BLEU_CALCULATOR_PATH = os.path.join(CURRENT_DIR_PATH, 'multi-bleu.perl')


class Mask:
    @classmethod
    def create_padding_mask(cls, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    @classmethod
    def create_look_ahead_mask(cls, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    @classmethod
    def create_masks(cls, inputs, target):
        encoder_padding_mask = Mask.create_padding_mask(inputs)
        decoder_padding_mask = Mask.create_padding_mask(inputs)

        look_ahead_mask = tf.maximum(
            Mask.create_look_ahead_mask(tf.shape(target)[1]),
            Mask.create_padding_mask(target)
        )

        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.lr = 0

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        self.lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return self.lr


def label_smoothing(target_data, depth, epsilon=0.1):
    target_data_one_hot = tf.one_hot(target_data, depth=depth)
    n = target_data_one_hot.get_shape().as_list()[-1]
    return ((1 - epsilon) * target_data_one_hot) + (epsilon / n)


class Trainer:
    def __init__(
            self,
            model,
            dataset,
            loss_object=None,
            optimizer=None,
            checkpoint_dir='./checkpoints',
            batch_size=None,
            distribute_strategy=None,
            vocab_size=32000,
            epoch=20,
    ):
        self.batch_size = batch_size
        self.distribute_strategy = distribute_strategy
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.vocab_size = vocab_size
        self.epoch = epoch
        self.dataset = dataset

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.optimizer is None:
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        else:
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')

    def multi_gpu_train(self, reset_checkpoint=False):
        with self.distribute_strategy.scope():
            self.dataset = self.distribute_strategy.experimental_distribute_dataset(self.dataset)
            self.trainer(reset_checkpoint=reset_checkpoint, is_distributed=True)

    def single_gpu_train(self, reset_checkpoint=False):
        self.trainer(reset_checkpoint=reset_checkpoint, is_distributed=False)

    def trainer(self, reset_checkpoint, is_distributed=False):
        current_day = datetime.datetime.now().strftime("%Y%m%d")
        train_log_dir = './logs/gradient_tape/' + current_day + '/train'
        os.makedirs(train_log_dir, exist_ok=True)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if not reset_checkpoint:
            if self.checkpoint_manager.latest_checkpoint:
                print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

            self.checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint
            )
        else:
            print("reset and initializing from scratch.")

        for epoch in range(self.epoch):
            start = time.time()
            print('start learning')

            for (batch, (inputs, target)) in enumerate(self.dataset):
                if is_distributed:
                    self.distributed_train_step(inputs, target)
                else:
                    self.train_step(inputs, target)

                self.checkpoint.step.assign_add(1)
                if batch % 50 == 0:
                    print(
                        "Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}".format(epoch, batch, self.train_loss.result(),
                                                                              self.train_accuracy.result()))
                if batch % 10000 == 0 and batch != 0:
                    self.checkpoint_manager.save()
            print("{} | Epoch: {} Loss:{}, Accuracy: {}, time: {} sec".format(
                datetime.datetime.now(), epoch, self.train_loss.result(), self.train_accuracy.result(),
                time.time() - start
            ))
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step=epoch)

            self.checkpoint_manager.save()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()
        self.checkpoint_manager.save()

    def basic_train_step(self, inputs, target):
        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            inputs, target_include_start
        )

        with tf.GradientTape() as tape:
            pred = self.model.call(
                inputs=inputs,
                target=target_include_start,
                inputs_padding_mask=encoder_padding_mask,
                look_ahead_mask=look_ahead_mask,
                target_padding_mask=decoder_padding_mask,
                training=True
            )

            loss = self.loss_function(target_include_end, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(target_include_end, pred)

        if self.distribute_strategy is None:
            return tf.reduce_mean(loss)

        return loss

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        real_one_hot = label_smoothing(real, depth=self.vocab_size)
        loss = self.loss_object(real_one_hot, pred)

        mask = tf.cast(mask, dtype=loss.dtype)

        loss *= mask
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, inputs, target):
        return self.basic_train_step(inputs, target)

    @tf.function
    def distributed_train_step(self, inputs, target):
        loss = self.distribute_strategy.experimental_run_v2(self.basic_train_step, args=(inputs, target))
        loss_value = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        return tf.reduce_mean(loss_value)


def translate(inputs, data_loader, model, seq_max_len_target=100):
    if data_loader is None:
        ValueError('data loader is None')

    if model is None:
        ValueError('model is None')

    if not isinstance(seq_max_len_target, int):
        ValueError('seq_max_len_target is not int')

    if isinstance(inputs, str):
        inputs = [inputs]

    encoded_data = []
    for sentence in inputs:
        d = data_loader.encode_data(sentence, mode='source')
        encoded_data.append(d)
    encoded_data = data_loader.texts_to_sequences(encoded_data)
    # pad
    encoded_data = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_data, maxlen=seq_max_len_target, padding='post'
    )
    encoder_inputs = tf.convert_to_tensor(encoded_data, dtype=tf.int32)

    batch_size = encoder_inputs.shape[0]
    decoder_inputs = [data_loader.dictionary['target']['token2idx']['<s>']] * batch_size
    decoder_inputs = tf.expand_dims(decoder_inputs, 1)
    decoder_end_token = data_loader.dictionary['target']['token2idx']['</s>']

    maxlen = min(seq_max_len_target, 2 * encoder_inputs.shape[1])

    for i in range(maxlen):
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            encoder_inputs, decoder_inputs
        )
        pred = model.call(
            [encoder_inputs,
             decoder_inputs,
             encoder_padding_mask,
             look_ahead_mask,
             decoder_padding_mask,
             ],
            training=False
        )
        pred = pred[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)
        decoder_inputs = tf.concat([decoder_inputs, predicted_id], axis=-1)

        tf.keras.backend.clear_session()
        gc.collect()

    total_output = [d[:d.index(decoder_end_token) + 1] if decoder_end_token in d else d
                    for d in decoder_inputs.numpy().tolist()]
    total_output = data_loader.sequences_to_texts(total_output, mode='target')
    return total_output


def translate_keras_sampler(inputs, data_loader, model, seq_max_len_target=100, num_beams=3, sampler_name='beam'):
    if data_loader is None:
        ValueError('data loader is None')

    if model is None:
        ValueError('model is None')

    if not isinstance(seq_max_len_target, int):
        ValueError('seq_max_len_target is not int')

    if isinstance(inputs, str):
        inputs = [inputs]

    encoded_data = []
    for sentence in inputs:
        d = data_loader.encode_data(sentence, mode='source')
        encoded_data.append(d)
    encoded_data = data_loader.texts_to_sequences(encoded_data)

    # pad
    encoded_data = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_data, maxlen=seq_max_len_target, padding='post'
    )
    encoder_inputs = tf.convert_to_tensor(encoded_data, dtype=tf.int32)

    batch_size = encoder_inputs.shape[0]
    decoder_end_token = data_loader.dictionary['target']['token2idx']['</s>']
    maxlen = min(seq_max_len_target, 2 * encoder_inputs.shape[1])
    prompt = np.full((batch_size, maxlen), data_loader.dictionary['target']['token2idx']['<s>'], dtype="int32")

    if sampler_name == 'beam':
        encoder_inputs = tf.repeat(
            encoder_inputs,
            repeats=num_beams,
            axis=0
        )

    sampler = None
    print('sampler_name', sampler_name)
    sampler_kwargs = {}
    if sampler_name == 'beam':
        sampler = keras_nlp.samplers.BeamSampler(num_beams=num_beams)
    elif sampler_name == 'contrastive':
        sampler = keras_nlp.samplers.ContrastiveSampler()
        hs = np.ones([batch_size, 1, 512], dtype="int32")
        sampler_kwargs.update({'hidden_states': hs})
    elif sampler_name == 'topk':
        sampler = keras_nlp.samplers.TopKSampler(k=3)
    elif sampler_name == 'topp':
        sampler = keras_nlp.samplers.TopPSampler(p=0.1)
    else:
        raise NotImplementedError

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(dec, cache, index):
        print(encoder_inputs.dtype)
        print(dec.dtype)
        print('prompt', prompt.dtype, hs.dtype, decoder_end_token)
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            encoder_inputs, dec
        )

        print(encoder_padding_mask.dtype, look_ahead_mask.dtype, decoder_padding_mask.dtype)
        pred = model(
            [encoder_inputs,
             dec,
             encoder_padding_mask,
             look_ahead_mask,
             decoder_padding_mask,
             ],
            training=False
        )[:, index - 1, :]
        # print(dec.shape)
        return pred, None, cache

    # Print the generated sequence (token ids).
    output = sampler(
        token_probability_fn,
        prompt=prompt,
        end_token_id=decoder_end_token,
        **sampler_kwargs
    )

    total_output = [d[:d.index(decoder_end_token) + 1] if decoder_end_token in d else d
                    for d in output.numpy().tolist()]
    total_output = data_loader.sequences_to_texts(total_output, mode='target')
    return total_output


def calculate_bleu_score(target_path, ref_path):
    get_bleu_score = f"perl {BLEU_CALCULATOR_PATH} {ref_path} < {target_path} > temp"
    print(get_bleu_score)
    os.system(get_bleu_score)
    with open("temp", "r") as f:
        bleu_score_report = f.read()
    print(bleu_score_report)
    score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]

    return score, bleu_score_report
