
# Anthe

Anthe is an architecture that achieves better performance than Transformer with less parameters.
This is the official repository for the article [Less is More!
A slim architecture for optimal language translation](https://arxiv.org/pdf/2305.10991.pdf), recently submitted 
to NeurIPS 2023.

To run the experiments run the ```main.py``` file. If you want to activate the Transformer architecture, pass the 
argument ```--comments=sameemb_projectoutput```. If you want to activate the Anthe architecture, pass the argument 
```--comments=geglu_gateattention_hsoftpos:2_tcffn:.005_tcpreatt:.07_tclength:2```. By default it will use
the WMT14 dataset. If you want to use the WMT17 add the following text to the comments argument: 
```--comments=..._lpair:cs-en```, where the available 
language pairs are cs-en, de-en, fi-en, lv-en, ru-en, tr-en, zh-en.


### Acknowledgements
We thank [strutive07](https://github.com/strutive07/transformer-tensorflow2.0) for the code of the Transformer and 
WMT14 task, which we used as a starting point for our code.
