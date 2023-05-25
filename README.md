# Anthe

This is the official repository for the article [Less is More!
A slim architecture for optimal language translation](https://arxiv.org/pdf/2305.10991.pdf). Anthe is an architecture
that improves on the Transformer performance with much fewer parameters.

To run the experiments run the ```train.py``` file. If you want to activate the Transformer architecture, pass the
argument ```--comments=sameemb_projectoutput```. If you want to activate the Anthe architecture, pass the argument
```--comments=geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2```. By default it will use
the WMT14 dataset. If you want to use the WMT17 add the following text to the comments argument:
```--comments=..._lpair:cs-en```, where the available
language pairs are cs-en, de-en, fi-en, lv-en, ru-en, tr-en, zh-en.

## Layers Available

The following layers are available for the Anthe architecture, only in TensorFlow 2.10.0 for now. 
You can access the Anthe architecture, the AntheEncoderBlock and the AntheDecoderBlock, like so:

```python
from anthe_official.neural_models import Anthe, AntheEncoderBlock, AntheDecoderBlock

model = Anthe(
    inputs_vocab_size, target_vocab_size, encoder_count, decoder_count, attention_head_count,
    d_model, d_point_wise_ff, dropout_prob
)

encoder_block = AntheEncoderBlock(
    attention_head_count, d_model, d_point_wise_ff, dropout_prob
)

decoder_block = AntheDecoderBlock(
    attention_head_count, d_model, d_point_wise_ff, dropout_prob
)
```

In the article we develop other layers that are part of the Anthe architecture, but might be of interest
on their own.
The TC versions of the Dense, Conv1D and Embedding,
and the SoftPOS and the HSoftPOS, can be accessed like so:

```python
from anthe_official.neural_models import *

tc_dense = TCDense(d_model, length=3, ratio=.2)
tc_conv1d = TCConv1D(filters, kernel_size, tc_length=3, ratio=.2)
tc_embedding = TCEmbedding(input_dim, output_dim, tc_length=3, ratio=.2)

soft_pos = SoftPOS(add_units, n_subpos=add_units, repeat_subpos=1)
hsoft_pos = HSoftPOS(vocab_size, embed_dim)
```


### Acknowledgements

We thank [strutive07](https://github.com/strutive07/transformer-tensorflow2.0)
for his implementation of the Transformer and
WMT14 task, which we used as a starting point for our code.
