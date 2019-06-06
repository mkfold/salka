![](https://repository-images.githubusercontent.com/184971303/09029180-87e3-11e9-8633-cfc73a9642d3)


neural networks for unsupervised anomaly detection in computer networks. written in pytorch.

based on/inspired by the work of [Tuor et al.](https://arxiv.org/abs/1712.00557) and [safekit](https://github.com/pnnl/safekit), this project sets out to apply more recent approaches to deep language modeling using more recent frameworks.

## features
* recurrent neural network language models (uni/bidirectional)
* transformer networks, based on [GPT](https://openai.com/blog/language-unsupervised/)/[GPT-2](https://openai.com/blog/better-language-models/)
* dataset tools:
  * parse and tokenize log files into character-level features on-the-fly

## dependencies
* python 3.7
* pytorch
* numpy
* click
* tqdm

if you use pyenv+pipenv, there's a Pipfile to get you started :sparkles:

## todo
* analysis tools:
  * evaluate model performance if labels are available
* dataset tools:
  * support for really large text files
  * support for byte pair encoding, word-level encoding

## references
* Recurrent language model: 
  * [Recurrent Neural Network Language Models for Open Vocabulary Event-Level Cyber Anomaly Detection](https://arxiv.org/abs/1712.00557)
* Transformer language model:
  * [Improving Language Understanding with Unsupervised Learning](https://openai.com/blog/language-unsupervised/) and accompanying paper, "Improving Language Understanding by Generative Pre-Training"
  * [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/) and accompanying paper, "Language Models are Unsupervised Multitask Learners"
* Byte pair encoding: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
