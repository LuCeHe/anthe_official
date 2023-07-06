from __future__ import (absolute_import, division, print_function, unicode_literals)
import tensorflow as tf
import os, argparse, random, socket, time, json, shutil, sys

sys.path.append('../')

from pyaromatics.keras_tools.esoteric_callbacks.several_validations import MultipleValidationSets
from pyaromatics.keras_tools.model_checkpoint import CustomModelCheckpoint
from pyaromatics.keras_tools.silence_tensorflow import silence_tf

silence_tf()

import numpy as np
import pandas as pd

from pyaromatics.keras_tools.esoteric_callbacks import LearningRateLogger, TimeStopping, CSVLogger
from pyaromatics.keras_tools.esoteric_losses import sparse_perplexity
from pyaromatics.keras_tools.plot_tools import plot_history
from pyaromatics.stay_organized.utils import NumpyEncoder, str2val
from pyaromatics.keras_tools.esoteric_callbacks import ClearMemory
from anthe_official.generation_data.data_loader import WMT_ENDE
from anthe_official.generation_data.wmt17 import WMT17
from anthe_official.neural_models_tf.transformer import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'wmt'))
os.makedirs(DATADIR, exist_ok=True)

EXPERIMENTS = os.path.join(CDIR, 'experiments')
CKPT_BEST = os.path.join(CDIR, 'ckpt_best')
CKPT = os.path.join(CDIR, 'ckpt')
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
random_string = ''.join([str(r) for r in np.random.choice(10, 5)])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comments",
        default='layerhspos:2',
        type=str, help="String to activate extra behaviors")
    parser.add_argument("--seed", default=39, type=int, help="Random seed")
    parser.add_argument("--epochs", default=3, type=int, help="Epochs")
    parser.add_argument("--steps_per_epoch", default=1, type=int, help="Steps per epoch")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--stop_time", default=2000, type=int, help="Stop time")
    parser.add_argument("--d_model", default=512, type=int, help="Model width")
    parser.add_argument("--results_dir", default=EXPERIMENTS, type=str, help="Experiments Folder")
    parser.add_argument("--checkpoint_dir", default=CKPT, type=str, help="Checkpoints Folder")
    parser.add_argument("--checkpoint_period", default=5, type=int,
                        help="Checkpoint frequency in epochs, if <=0, no saving")
    parser.add_argument("--save_best", default=1, type=int, help="whether to save best model 1/0 -> True/False")
    args = parser.parse_args()

    name_exp = time_string + random_string + '--antheofficial'
    args.experiment_dir = os.path.join(args.results_dir, name_exp)
    os.makedirs(args.experiment_dir, exist_ok=True)

    args.ckpt_dir = os.path.join(args.checkpoint_dir, name_exp + '--ckpt')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    args.ckpt_best_dir = os.path.join(CKPT_BEST, name_exp + '--ckpt-best')
    os.makedirs(args.ckpt_best_dir, exist_ok=True)

    return args


def main(args):
    comments = args.comments
    results = vars(args)
    print(json.dumps(vars(args), indent=4, cls=NumpyEncoder))

    experiment_dir = args.experiment_dir
    ckpt_dir = args.ckpt_dir
    ckpt_best_dir = args.ckpt_best_dir

    print(experiment_dir)
    print(ckpt_dir)
    print(ckpt_best_dir)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # hyper paramaters
    nlayers = str2val(args.comments, 'nlayers', int, default=6)

    TRAIN_RATIO = 0.9
    D_POINT_WISE_FF = 4 * args.d_model
    D_MODEL = args.d_model
    ENCODER_COUNT = DECODER_COUNT = nlayers
    ATTENTION_HEAD_COUNT = 8
    DROPOUT_PROB = 0.1
    SEQ_MAX_LEN_SOURCE = 100
    SEQ_MAX_LEN_TARGET = 101
    BPE_VOCAB_SIZE = 32000
    batch_frequency = 2000

    # 32,800,000 vs 20,586,880 (-16,416,000, /2)=8,192 vs 2,085

    # for overfitting test hyper parameters
    # BATCH_SIZE = 32
    # EPOCHS = 100
    DATA_LIMIT = None

    GLOBAL_BATCH_SIZE = (args.batch_size * 1)

    if 'lpair' in comments:
        lpair = str2val(args.comments, 'lpair', str, default='cs-en')
        maxlen = str2val(args.comments, 'maxlen', int, default=256)
        generator = lambda data_split: \
            WMT17(lpair, args.batch_size, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                  data_split=data_split, comments=comments, maxlen=maxlen)

        gen_train = generator('train')
        gen_val = generator('validation')
        gen_test = generator('test')
    else:
        generator = lambda data_split: \
            WMT_ENDE(
                data_dir=DATADIR, batch_size=GLOBAL_BATCH_SIZE, bpe_vocab_size=BPE_VOCAB_SIZE,
                seq_max_len_source=SEQ_MAX_LEN_SOURCE, seq_max_len_target=SEQ_MAX_LEN_TARGET, data_limit=DATA_LIMIT,
                train_ratio=TRAIN_RATIO, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                data_split=data_split,
                comments=comments)

        gen_train = generator('train')
        gen_train.produce_masks = True
        gen_val = generator('val')
        gen_val.produce_masks = True

    model = build_model(
        inputs_timesteps=SEQ_MAX_LEN_SOURCE,
        target_timesteps=SEQ_MAX_LEN_TARGET,
        inputs_vocab_size=BPE_VOCAB_SIZE,
        target_vocab_size=BPE_VOCAB_SIZE,
        encoder_count=ENCODER_COUNT,
        decoder_count=DECODER_COUNT,
        attention_head_count=ATTENTION_HEAD_COUNT,
        d_model=D_MODEL,
        d_point_wise_ff=D_POINT_WISE_FF,
        dropout_prob=DROPOUT_PROB,
        comments=args.comments)

    results['n_params'] = model.count_params()
    save_results(results, args.experiment_dir)

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

    learning_rate = str2val(args.comments, 'lr', float, default=3.16e-5)
    optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    eager = True if 'eager' in comments else False
    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[
            'sparse_categorical_accuracy', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            sparse_perplexity
        ],
        run_eagerly=eager)
    callbacks = [
        LearningRateLogger(),
        ClearMemory(batch_frequency=batch_frequency, verbose=0),
        TimeStopping(args.stop_time, 1, stop_within_epoch=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
    if args.save_best > 0:
        model_checkpoint_best = CustomModelCheckpoint(
            filepath=ckpt_best_dir + '/' + 'best_model.h5',
            save_weights_only=True,
            monitor='val_loss',
            # 'val_sparse_perplexity', 'val_sparse_categorical_accuracy' 'val_sparse_categorical_crossentropy'
            mode='min',
            save_best_only=True,
            verbose=0)
        callbacks.append(model_checkpoint_best)

    if args.checkpoint_period > 0:
        model_checkpoint = CustomModelCheckpoint(
            filepath=ckpt_dir + '/' + 'last_model.h5',
            save_weights_only=False,
            save_best_only=False,
            verbose=0,
            save_freq="period", period=args.checkpoint_period)
        callbacks.append(model_checkpoint)

    history_path = os.path.join(experiment_dir, 'history.csv')

    if 'lpair' in comments:
        callbacks.append(MultipleValidationSets({'t': gen_test}, verbose=1), )

    callbacks.append(CSVLogger(history_path), )
    # notice that the wmt dataloader does not work with shuffle=True
    shuffle = False if 'lpair' in comments else True
    model.fit(gen_train, validation_data=gen_val, callbacks=callbacks, epochs=args.epochs, shuffle=shuffle)

    if args.epochs > 0:
        history_df = pd.read_csv(history_path)

        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}
        json_filename = os.path.join(experiment_dir, 'history.json')
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

        history_keys = history_df.columns.tolist()
        lengh_keys = 6
        no_vals_keys = [k for k in history_keys if not k.startswith('val_')]
        all_chunks = [no_vals_keys[x:x + lengh_keys] for x in range(0, len(no_vals_keys), lengh_keys)]
        for i, subkeys in enumerate(all_chunks):
            history_dict = {k: history_df[k].tolist() for k in subkeys}
            history_dict.update(
                {'val_' + k: history_df['val_' + k].tolist() for k in subkeys if 'val_' + k in history_keys})
            plot_filename = os.path.join(experiment_dir, f'history_{i}.png')
            plot_history(histories=history_dict, plot_filename=plot_filename, epochs=args.epochs)

    print('Evaluating on validation and test set...')
    for data_split in ['validation', 'test']:
        try:
            gen = generator(data_split)
            gen.produce_masks = True
            evaluation = model.evaluate(gen, return_dict=True)
            for k in evaluation.keys():
                results[f'{data_split}_{k}'] = evaluation[k]
        except Exception as e:
            print(f'Error while evaluating on {data_split}: {e}')

    return results


def save_results(results, experiment_dir):
    with open(os.path.join(experiment_dir, 'results.txt'), "w") as f:
        f.write(json.dumps(results, indent=4, cls=NumpyEncoder))


if __name__ == "__main__":
    args = get_args()

    save_results(vars(args), args.experiment_dir)

    time_start = time.perf_counter()
    results = main(args)
    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in ' + str(time_elapsed) + 's')

    results.update(time_elapsed=time_elapsed)
    results.update(hostname=socket.gethostname())

    string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
    print(string_result)
    path = os.path.join(args.experiment_dir, 'results.txt')
    with open(path, "w") as f:
        f.write(string_result)

    shutil.make_archive(args.experiment_dir, 'zip', args.experiment_dir)
