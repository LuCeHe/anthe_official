import os, shutil

import tensorflow as tf
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset_builder
import numpy as np
from datasets import load_from_disk, load_dataset

from anthe_official.generation_data.utils import Mask

from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE

data_links = ['https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip']

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', 'data'))

WKT103DIR = os.path.join(DATADIR, 'wikitext103')
tokenizer_path = os.path.join(WKT103DIR, 'tokenizer-wiki.json')
os.makedirs(WKT103DIR, exist_ok=True)

pairs = ['cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']


def get_tokenizer():
    if not os.path.exists(tokenizer_path):
        from pyaromatics.stay_organized.download_utils import download_and_unzip
        import shutil
        if len(os.listdir(WKT103DIR)) == 0:
            download_and_unzip(data_links, WKT103DIR)

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=32000)
        tokenizer.pre_tokenizer = Whitespace()

        files = [os.path.join(WKT103DIR, 'wikitext-103-raw', f"wiki.{split}.raw")
                 for split in ["test", "train", "valid"]]
        tokenizer.train(files, trainer)

        tokenizer.save(tokenizer_path)

        shutil.rmtree(os.path.join(WKT103DIR, 'wikitext-103-raw'))


def tokenize_wmt17(sub_pairs = ['ru-en']):
    get_tokenizer()

    for langs in sub_pairs:
        l1, l2 = langs.split('-')
        print(l1, l2)

        WMTDIR = os.path.join(DATADIR, langs)
        os.makedirs(WMTDIR, exist_ok=True)

        preprocessed_data_path = os.path.join(WMTDIR, langs)

        print(preprocessed_data_path)

        if len(os.listdir(WMTDIR)) == 0:
            cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
            # wmt20_mlqe_task1 wmt17
            builder = load_dataset_builder('wmt17', langs, revision="main")

            print(builder.info.description)
            print(builder.info.features)
            builder.download_and_prepare()
            ds = builder.as_dataset()

            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            def bt(l1, l2, tokenizer):
                def batch_tokenization(batch):
                    lang1 = [d[l1] for d in batch['translation']]
                    lang2 = [d[l2] for d in batch['translation']]

                    try:
                        a = tokenizer(lang1, padding=True)['input_ids']
                        b = tokenizer(lang2, padding=True)['input_ids']
                        error = [[0]] * len(a)

                    except Exception as e:
                        a, b = [[0]] * len(lang2), [[0]] * len(lang2)
                        error = [[1]] * len(lang2)

                    return {l1: a, l2: b, 'error': error}

                return batch_tokenization

            ds = ds.map(bt(l1, l2, tokenizer),
                        remove_columns=["translation"], batched=True, batch_size=16, num_proc=4)

            ds = ds.filter(lambda example: example['error'] == [0])

            ds = ds.map(lambda example: example, remove_columns=['error'])

            if langs in ['ru-en', 'zh-en']:
                # ds = ds.filter(lambda example, indice: indice % 7 == 0, with_indices=True)
                # de-en 5906184
                ds = ds.filter(lambda example, indice: indice < 7000000, with_indices=True)

            ds.save_to_disk(preprocessed_data_path)

            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)

            print(ds['train'][0])
            print(ds['train'][:16])

            batch_size = 16

            batch = np.array(ds['train'][:batch_size][l1])
            print(batch.shape)


class WMT17(tf.keras.utils.Sequence):
    def __init__(self, language_pair, batch_size=16, epochs=1, steps_per_epoch=1, data_split='train', maxlen=256,
                 comments=''):

        assert language_pair in pairs
        assert data_split in ['train', 'validation', 'test']

        self.__dict__.update(
            language_pair=language_pair, epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size,
            data_split=data_split, comments=comments,
        )

        self.maxlen = maxlen
        self.epochs = 50 if epochs == None else epochs
        self.comments = comments

        WMTDIR = os.path.join(DATADIR, language_pair)
        self.preprocessed_data_path = os.path.join(WMTDIR, language_pair)
        self.on_epoch_end()
        self.l1, self.l2 = language_pair.split('-')

        if not os.path.exists(self.preprocessed_data_path):
            tokenize_wmt17(sub_pairs=[language_pair])
        dataset = load_from_disk(self.preprocessed_data_path)[self.data_split]
        n_samples = len(dataset)
        print(f"Loaded {n_samples} samples from {self.preprocessed_data_path} for {self.data_split} split")
        del dataset

        # load a multiple of the batchsize that is closer below to 10K, to avoid saturating the RAM
        max_samples = 5000 if self.data_split == 'train' else n_samples
        self.RAM_samples = int(max_samples / self.batch_size) * self.batch_size
        assert self.RAM_samples > 1
        self.dsf_l1 = None
        self.dsf_l2 = None
        self.data_split = data_split
        self.steps_per_epoch = int(n_samples / self.batch_size) - 1 \
            if self.steps_per_epoch < 0 else self.steps_per_epoch

        if 'pytorch' in comments:
            import torch

            self.pkg = torch
        else:
            import tensorflow as tf

            self.pkg = tf
    def get_fraction_of_dataset(self, index):

        if (index * self.batch_size) % self.RAM_samples == 0:
            fidx = index // self.RAM_samples

            dataset = load_from_disk(self.preprocessed_data_path, keep_in_memory=False)[self.data_split]
            dataset = dataset.select(range(fidx * self.RAM_samples, (fidx + 1) * self.RAM_samples)).shuffle(seed=42)

            del self.dsf_l1, self.dsf_l2
            self.dsf_l1 = np.array(dataset[self.l1])
            self.dsf_l2 = np.array(dataset[self.l2])

            del dataset

    def data_generation(self, index=None):
        i = (index * self.batch_size) % self.RAM_samples
        j = i + self.batch_size

        self.get_fraction_of_dataset(index)

        if not 'test' in self.comments:
            inputs = self.dsf_l1[i:j]
            target = self.dsf_l2[i:j]
        else:
            inputs = np.random.choice(3, (self.batch_size, self.seq_max_len_source))
            target = np.random.choice(3, (self.batch_size, self.seq_max_len_target))

        if len(target.shape) == 1:
            if 'pytorch' in self.comments:
                target = self.pkg.nn.functional.pad_sequence(target, padding_value=0, batch_first=True)
            else:
                target = self.pkg.keras.utils.pad_sequences(target.tolist(), padding='post', dtype="float32")

        if len(inputs.shape) == 1:
            if 'pytorch' in self.comments:
                inputs = self.pkg.nn.functional.pad_sequence(inputs, padding_value=0, batch_first=True)
            else:
                inputs = self.pkg.keras.utils.pad_sequences(inputs.tolist(), padding='post', dtype="float32")

        if not inputs.shape[1] + 1 == target.shape[1]:
            if inputs.shape[1] + 1 > target.shape[1]:
                l = inputs.shape[1] + 1

                if 'pytorch' in self.comments:
                    target = self.pkg.nn.functional.pad_sequence(target, padding_value=0, batch_first=True,
                                                                  maxlen=l)
                else:
                    target = self.pkg.keras.utils.pad_sequences(target.tolist(), padding='post', dtype="float32",
                                                                 maxlen=l)
            else:
                l = target.shape[1] - 1
                if 'pytorch' in self.comments:
                    inputs = self.pkg.nn.functional.pad_sequence(inputs, padding_value=0, batch_first=True, maxlen=l)
                else:
                    inputs = self.pkg.keras.utils.pad_sequences(inputs.tolist(), padding='post', dtype="float32",
                                                                 maxlen=l)

        if inputs.shape[1] > self.maxlen:
            inputs = inputs[:, :self.maxlen]
            target = target[:, :self.maxlen + 1]

        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            inputs, target_include_start
        )

        return dict(
            inputs=inputs, target_include_start=target_include_start, encoder_padding_mask=encoder_padding_mask,
            look_ahead_mask=look_ahead_mask, decoder_padding_mask=decoder_padding_mask,
            target_include_end=target_include_end
        )

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation(index)
        return [
                   batch['inputs'],
                   batch['target_include_start'],
                   batch['encoder_padding_mask'],
                   batch['look_ahead_mask'],
                   batch['decoder_padding_mask']
               ], batch['target_include_end']


if __name__ == '__main__':
    tokenize_wmt17()
    gen = WMT17('cs-en', 16, epochs=1, steps_per_epoch=1, data_split='train', comments='')
    batch = gen.__getitem__()
    print(batch)
