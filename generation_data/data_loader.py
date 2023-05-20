import os, copy, pickle
from urllib.request import urlretrieve
import numpy as np
import sentencepiece
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py as hp

from anthe_official.generation_data.utils import Mask


class DataLoader:
    DIR = None
    PATHS = {}
    BPE_VOCAB_SIZE = 0
    MODES = ['source', 'target']
    dictionary = {
        'source': {
            'token2idx': None,
            'idx2token': None,
        },
        'target': {
            'token2idx': None,
            'idx2token': None,
        }
    }
    CONFIG = {
        'wmt14/en-de': {
            'source_lang': 'en',
            'target_lang': 'de',
            'base_url': 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/',
            'train_files': ['train.en', 'train.de'],
            'vocab_files': ['vocab.50K.en', 'vocab.50K.de'],
            'dictionary_files': ['dict.en-de'],
            'test_files': [
                'newstest2012.en', 'newstest2012.de',
                'newstest2013.en', 'newstest2013.de',
                'newstest2014.en', 'newstest2014.de',
                'newstest2015.en', 'newstest2015.de',
            ]
        }
    }
    BPE_MODEL_SUFFIX = '.model'
    BPE_VOCAB_SUFFIX = '.vocab'
    BPE_RESULT_SUFFIX = '.sequences'
    SEQ_MAX_LEN = {
        'source': 100,
        'target': 100
    }
    DATA_LIMIT = None
    TRAIN_RATIO = 0.9
    BATCH_SIZE = 16

    source_sp = None
    target_sp = None

    def __init__(self, dataset_name, data_dir, batch_size=16, bpe_vocab_size=32000, seq_max_len_source=100,
                 seq_max_len_target=100, data_limit=None, train_ratio=0.9):
        if dataset_name is None or data_dir is None:
            raise ValueError('dataset_name and data_dir must be defined')
        self.DIR = data_dir
        self.DATASET = dataset_name
        self.BPE_VOCAB_SIZE = bpe_vocab_size
        self.SEQ_MAX_LEN['source'] = seq_max_len_source
        self.SEQ_MAX_LEN['target'] = seq_max_len_target
        self.DATA_LIMIT = data_limit
        self.TRAIN_RATIO = train_ratio
        self.BATCH_SIZE = batch_size

        self.PATHS['source_data'] = os.path.join(self.DIR, self.CONFIG[self.DATASET]['train_files'][0])
        self.PATHS['source_bpe_prefix'] = self.PATHS['source_data'].replace('train', 'bpe_train') + '.segmented'

        self.PATHS['saved_train'] = os.path.join(self.DIR, 'saved_train.h5')
        self.PATHS['saved_val'] = os.path.join(self.DIR, 'saved_val.h5')
        self.PATHS['target_data'] = os.path.join(self.DIR, self.CONFIG[self.DATASET]['train_files'][1])
        self.PATHS['target_bpe_prefix'] = self.PATHS['target_data'].replace('train', 'bpe_train') + '.segmented'

    def load(self, custom_dataset=False):

        if not os.path.exists(self.PATHS['saved_train']):

            if custom_dataset:
                print('#1 use custom dataset. please implement custom download_dataset function.')
            else:
                print('#1 download data')
                self.download_dataset()

            print('#2 parse data')
            target_data = self.parse_data_and_save(self.PATHS['target_data'])
            source_data = self.parse_data_and_save(self.PATHS['source_data'])

            print('#3 train bpe')

            self.train_bpe(self.PATHS['source_data'], self.PATHS['source_bpe_prefix'])
            self.train_bpe(self.PATHS['target_data'], self.PATHS['target_bpe_prefix'])

            print('#4 load bpe vocab')
            # print(self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)
            # print(self.PATHS['source_bpe_prefix'])

            self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] = self.load_bpe_vocab(
                self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)
            self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
                self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)

            print('#5 encode data with bpe')
            source_sequences = self.texts_to_sequences(
                self.sentence_piece(
                    source_data,
                    self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX,
                    self.PATHS['source_bpe_prefix'] + self.BPE_RESULT_SUFFIX
                ),
                mode="source"
            )
            target_sequences = self.texts_to_sequences(
                self.sentence_piece(
                    target_data,
                    self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX,
                    self.PATHS['target_bpe_prefix'] + self.BPE_RESULT_SUFFIX
                ),
                mode="target"
            )

            print('source sequence example:', source_sequences[0])
            print('target sequence example:', target_sequences[0])
            print('source samples:         ', len(source_sequences))
            print('target samples:         ', len(target_sequences))

            if self.TRAIN_RATIO == 1.0:
                source_sequences_train = source_sequences
                source_sequences_val = []
                target_sequences_train = target_sequences
                target_sequences_val = []
            else:
                (source_sequences_train,
                 source_sequences_val,
                 target_sequences_train,
                 target_sequences_val) = train_test_split(
                    source_sequences, target_sequences, train_size=self.TRAIN_RATIO
                )

            if self.DATA_LIMIT is not None:
                print('data size limit ON. limit size:', self.DATA_LIMIT)
                source_sequences_train = source_sequences_train[:self.DATA_LIMIT]
                target_sequences_train = target_sequences_train[:self.DATA_LIMIT]

            print('source_sequences_train', len(source_sequences_train))
            print('source_sequences_val', len(source_sequences_val))
            print('target_sequences_train', len(target_sequences_train))
            print('target_sequences_val', len(target_sequences_val))

            print('train set size: ', len(source_sequences_train))
            print('validation set size: ', len(source_sequences_val))

            s_numpy_train, t_numpy_train = self.create_numpies(source_sequences_train, target_sequences_train)
            s_numpy_val, t_numpy_val = self.create_numpies(source_sequences_val, target_sequences_val)

            f = hp.File(self.PATHS['saved_train'], 'w')
            f.create_dataset('s_numpy_train', data=s_numpy_train)
            f.create_dataset('t_numpy_train', data=t_numpy_train)
            f.close()

            f = hp.File(self.PATHS['saved_val'], 'w')
            f.create_dataset('s_numpy_val', data=s_numpy_val)
            f.create_dataset('t_numpy_val', data=t_numpy_val)
            f.close()

        f = hp.File(self.PATHS['saved_train'], 'r')
        s_numpy_train = f['s_numpy_train'][:]
        t_numpy_train = f['t_numpy_train'][:]
        f.close()

        f = hp.File(self.PATHS['saved_val'], 'r')
        s_numpy_val = f['s_numpy_val'][:]
        t_numpy_val = f['t_numpy_val'][:]
        f.close()

        train_dataset = self.create_dataset(s_numpy_train, t_numpy_train)
        val_dataset = self.create_dataset(s_numpy_val, t_numpy_val)
        return train_dataset, val_dataset

    def load_test(self, index=0, custom_dataset=False):

        if index < 0 or index >= len(self.CONFIG[self.DATASET]['test_files']) // 2:
            raise ValueError('test file index out of range. min: 0, max: {}'.format(
                len(self.CONFIG[self.DATASET]['test_files']) // 2 - 1)
            )
        if custom_dataset:
            print('#1 use custom dataset. please implement custom download_dataset function.')
        else:
            print('#1 download data')
            self.download_dataset()

        print('#2 parse data')

        source_test_data_path, target_test_data_path = self.get_test_data_path(index)

        source_data = self.parse_data_and_save(source_test_data_path)
        target_data = self.parse_data_and_save(target_test_data_path)

        print('#3 load bpe vocab')

        self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX)

        return source_data, target_data

    def get_test_data_path(self, index):
        source_test_data_path = os.path.join(self.DIR, self.CONFIG[self.DATASET]['test_files'][index * 2])
        target_test_data_path = os.path.join(self.DIR, self.CONFIG[self.DATASET]['test_files'][index * 2 + 1])
        return source_test_data_path, target_test_data_path

    def download_dataset(self):
        for file in (self.CONFIG[self.DATASET]['train_files']
                     + self.CONFIG[self.DATASET]['vocab_files']
                     + self.CONFIG[self.DATASET]['dictionary_files']
                     + self.CONFIG[self.DATASET]['test_files']):
            self._download("{}{}".format(self.CONFIG[self.DATASET]['base_url'], file))

    def _download(self, url):
        path = os.path.join(self.DIR, url.split('/')[-1])
        if not os.path.exists(path):
            with TqdmCustom(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url) as t:
                urlretrieve(url, path, t.update_to)

    def parse_data_and_save(self, path):
        print('load data from {}'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        if lines is None:
            raise ValueError('Vocab file is invalid')
        return lines

    def train_bpe(self, data_path, model_prefix):
        # print(self.BPE_MODEL_SUFFIX)
        model_path = model_prefix + self.BPE_MODEL_SUFFIX
        vocab_path = model_prefix + self.BPE_VOCAB_SUFFIX

        if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
            print('bpe model does not exist. train bpe. model path:', model_path, ' vocab path:', vocab_path)
            train_source_params = "--inputs={} \
                --pad_id=0 \
                --unk_id=1 \
                --bos_id=2 \
                --eos_id=3 \
                --model_prefix={} \
                --vocab_size={} \
                --model_type=bpe ".format(
                data_path,
                model_prefix,
                self.BPE_VOCAB_SIZE
            )
            sentencepiece.SentencePieceTrainer.Train(train_source_params)
        else:
            print('bpe model exist. load bpe. model path:', model_path, ' vocab path:', vocab_path)

    def load_bpe_encoder(self):
        self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX
        )
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX
        )

    def sentence_piece(self, source_data, source_bpe_model_path, result_data_path):
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(source_bpe_model_path)

        if os.path.exists(result_data_path):
            print('encoded data exist. load data. path:', result_data_path)
            with open(result_data_path, 'r', encoding='utf-8') as f:
                sequences = f.read().strip().split('\n')
                return sequences

        print('encoded data does not exist. encode data. path:', result_data_path)
        sequences = []
        with open(result_data_path, 'w', encoding="utf8") as f:
            i = 0
            for sentence in tqdm(source_data):
                i += 1
                pieces = sp.EncodeAsPieces(sentence)
                sequence = " ".join(pieces)
                if not self.DATA_LIMIT is None and i > self.DATA_LIMIT:
                    break
                sequences.append(sequence)
                f.write(sequence + "\n")
        return sequences

    def encode_data(self, inputs, mode='source'):
        if mode not in self.MODES:
            ValueError('not allowed mode.')

        if mode == 'source':
            if self.source_sp is None:
                self.source_sp = sentencepiece.SentencePieceProcessor()
                self.source_sp.load(self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX)

            pieces = self.source_sp.EncodeAsPieces(inputs)
            sequence = " ".join(pieces)

        elif mode == 'target':
            if self.target_sp is None:
                self.target_sp = sentencepiece.SentencePieceProcessor()
                self.target_sp.load(self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX)

            pieces = self.target_sp.EncodeAsPieces(inputs)
            sequence = " ".join(pieces)

        else:
            ValueError('not allowed mode.')

        return sequence

    def load_bpe_vocab(self, bpe_vocab_path):
        with open(bpe_vocab_path, 'r', encoding="utf8") as f:
            vocab = [line.split()[0] for line in f.read().splitlines()]

        token2idx = {}
        idx2token = {}

        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token

    def texts_to_sequences(self, texts, mode='source'):
        if mode not in self.MODES:
            ValueError('not allowed mode.')

        sequences = []
        for text in texts:
            text_list = ["<s>"] + text.split() + ["</s>"]

            sequence = [
                self.dictionary[mode]['token2idx'].get(
                    token, self.dictionary[mode]['token2idx']["<unk>"]
                )
                for token in text_list
            ]
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences, mode='source'):
        if mode not in self.MODES:
            ValueError('not allowed mode.')

        texts = []
        for sequence in sequences:
            if mode == 'source':
                if self.source_sp is None:
                    self.source_sp = sentencepiece.SentencePieceProcessor()
                    self.source_sp.load(self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX)
                text = self.source_sp.DecodeIds(sequence)
            else:
                if self.target_sp is None:
                    self.target_sp = sentencepiece.SentencePieceProcessor()
                    self.target_sp.load(self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX)
                text = self.target_sp.DecodeIds(sequence)
            texts.append(text)
        return texts

    def create_numpies(self, source_sequences, target_sequences):
        new_source_sequences = []
        new_target_sequences = []
        for source, target in zip(source_sequences, target_sequences):
            if len(source) > self.SEQ_MAX_LEN['source']:
                continue
            if len(target) > self.SEQ_MAX_LEN['target']:
                continue
            new_source_sequences.append(source)
            new_target_sequences.append(target)

        source_numpy = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=new_source_sequences, maxlen=self.SEQ_MAX_LEN['source'], padding='post'
        )
        target_numpy = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=new_target_sequences, maxlen=self.SEQ_MAX_LEN['target'], padding='post'
        )
        return source_numpy, target_numpy

    def create_dataset(self, source_numpy, target_numpy):

        buffer_size = int(source_numpy.shape[0] * 0.3)
        dataset = tf.data.Dataset.from_tensor_slices(
            (source_numpy, target_numpy)
        ).shuffle(buffer_size)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


class TqdmCustom(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class WMT_ENDE(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, bpe_vocab_size, seq_max_len_source, seq_max_len_target, data_limit,
                 train_ratio, epochs=1, steps_per_epoch=1, data_split='train', comments=''):

        assert data_split in ['train', 'val', 'validation', 'test']

        self.__dict__.update(
            epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, data_dir=data_dir,
            bpe_vocab_size=bpe_vocab_size, seq_max_len_source=seq_max_len_source, seq_max_len_target=seq_max_len_target,
            data_limit=data_limit, train_ratio=train_ratio, data_split=data_split, comments=comments,
        )

        self.epochs = 50 if epochs == None else epochs

        if not 'test' in comments:
            self.data_loader = DataLoader(
                dataset_name='wmt14/en-de',
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                bpe_vocab_size=self.bpe_vocab_size,
                seq_max_len_source=self.seq_max_len_source,
                seq_max_len_target=self.seq_max_len_target,
                data_limit=self.data_limit,
                train_ratio=train_ratio
            )

            if self.data_split == 'train':
                self.dataset, _ = self.data_loader.load()

            elif self.data_split in ['validation', 'val']:
                _, self.dataset = self.data_loader.load()

            elif self.data_split == 'test':
                raise NotImplementedError  # _, self.dataset = self.data_loader.load_test(1)

            else:
                raise NotImplementedError

        self.on_epoch_end()
        self.produce_masks = False

    def on_epoch_end(self):
        if not 'test' in self.comments:
            n_samples = len(self.dataset)
            self.data_iterator = iter(self.dataset)
        else:
            self.data_iterator = None
            n_samples = 500

        self.steps_per_epoch = int(n_samples / self.batch_size) \
            if self.steps_per_epoch < 0 else self.steps_per_epoch

    def data_generation(self, index=None):

        if not 'test' in self.comments:
            inputs, target = next(self.data_iterator)
        else:
            inputs = np.random.choice(3, (self.batch_size, self.seq_max_len_source))
            target = np.random.choice(3, (self.batch_size, self.seq_max_len_target))

        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        outputs = dict(
            inputs=inputs, target_include_start=target_include_start,
            target_include_end=target_include_end
        )
        if self.produce_masks:
            encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
                inputs, target_include_start
            )
            outputs.update(encoder_padding_mask=encoder_padding_mask, look_ahead_mask=look_ahead_mask,
                           decoder_padding_mask=decoder_padding_mask)

        return outputs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation(index)
        if not self.produce_masks:
            return [
                   batch['inputs'],
                   batch['target_include_start'],
               ], batch['target_include_end']
        else:
            return [
                   batch['inputs'],
                   batch['target_include_start'],
                   batch['encoder_padding_mask'],
                   batch['look_ahead_mask'],
                   batch['decoder_padding_mask']
               ], batch['target_include_end']
