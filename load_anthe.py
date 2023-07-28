import json, os
import tensorflow as tf

from anthe_official.neural_models_tf import build_model
from anthe_official.generation_data.data_loader import WMT_ENDE
from pyaromatics.keras_tools.esoteric_losses import sparse_perplexity

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'wmt'))
os.makedirs(DATADIR, exist_ok=True)
ANTHE_DIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'anthe_models'))


# "comments": "geglu_gegluact:swish_gateattention_hsoftpos:2",
# -> sparse_perplexity:  vs 1.2633038759231567
# 1.3420/1.3420
config_path = os.path.join(ANTHE_DIR, '2023-07-07--16-19-29--01622--antheofficial', 'results.txt')
weights_path = os.path.join(ANTHE_DIR, '2023-07-07--16-19-29--01622--antheofficial--ckpt_best', 'best_model.h5')

# "comments": "geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2",
# -> sparse_perplexity:  vs 1.2670365571975708
# 1.3511/1.3511
config_path = os.path.join(ANTHE_DIR, '2023-07-07--16-19-38--54384--antheofficial', 'results.txt')
weights_path = os.path.join(ANTHE_DIR, '2023-07-07--16-19-38--54384--antheofficial--ckpt_best', 'best_model.h5')

# "comments": "geglu_gegluact:swish_gateattention_hsoftpos:2_tcffn:.1_tclength:2",
# -> sparse_perplexity:  vs 1.263352870941162,
# 1.3449/1.3449
config_path = os.path.join(ANTHE_DIR, '2023-07-07--16-24-26--76134--antheofficial', 'results.txt')
weights_path = os.path.join(ANTHE_DIR, '2023-07-07--16-24-26--76134--antheofficial--ckpt_best', 'best_model.h5')

# Read JSON file
with open(config_path) as data_file:
    results = json.load(data_file)

print(results)
string_result = json.dumps(results, indent=4)
print(string_result)

comments = results['comments']
d_model = results['d_model']

# hyper paramaters
nlayers = 6
steps_per_epoch = -1  # -1

TRAIN_RATIO = 0.9
D_POINT_WISE_FF = 4 * d_model
D_MODEL = d_model
ENCODER_COUNT = DECODER_COUNT = nlayers
ATTENTION_HEAD_COUNT = 8
DROPOUT_PROB = 0.0
SEQ_MAX_LEN_SOURCE = 100
SEQ_MAX_LEN_TARGET = 101
BPE_VOCAB_SIZE = 32000
batch_frequency = 2000
GLOBAL_BATCH_SIZE = (results['batch_size'] * 1)

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
    comments=comments
)

model.load_weights(weights_path)
model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='SGD',
    metrics=[
        'sparse_categorical_accuracy', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        sparse_perplexity
    ],
)

weights = model.get_weights()
print(len(weights))

print('Evaluating on validation and test set...')
generator = lambda data_split: \
    WMT_ENDE(
        data_dir=DATADIR, batch_size=GLOBAL_BATCH_SIZE, bpe_vocab_size=BPE_VOCAB_SIZE,
        seq_max_len_source=SEQ_MAX_LEN_SOURCE, seq_max_len_target=SEQ_MAX_LEN_TARGET, data_limit=None,
        train_ratio=TRAIN_RATIO, epochs=0, steps_per_epoch=steps_per_epoch,
        data_split=data_split, comments=comments
    )

for data_split in ['validation', 'test']:
    print('    ', data_split)
    try:
        gen = generator(data_split)
        gen.produce_masks = True
        evaluation = model.evaluate(gen, return_dict=True)

    except Exception as e:
        print(f'Error while evaluating on {data_split}: {e}')

print(f"  when in the experiment was {results['validation_sparse_perplexity']}")

import sys

sys.exit()

anthe_model = build_model(
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
    comments=comments)
anthe_model.load_weights(path)
anthe_model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='SGD',
    metrics=[
        'sparse_categorical_accuracy', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        sparse_perplexity
    ],
)

for data_split in ['validation', 'test']:
    print('    ', data_split)
    try:
        gen = generator(data_split)
        gen.produce_masks = True
        evaluation = anthe_model.evaluate(gen, return_dict=True)
    except Exception as e:
        print(f'Error while evaluating on {data_split}: {e}')

print('DONE')
