# salka
neural networks for unsupervised anomaly detection in computer networks. written in pytorch.

based on the work of [Tuor et al.](https://arxiv.org/abs/1712.00557), this project sets out to apply more recent approaches to deep language modeling using more recent frameworks.

## features
* recurrent neural network language models (uni/bidirectional)
* transformer networks, based on [GPT](https://openai.com/blog/language-unsupervised/)/[GPT-2](https://openai.com/blog/better-language-models/)
* dataset tools:
  * parse and tokenize log files into character-level features on-the-fly

## todo
* analysis tools:
  * generate scores based on labeled data
* dataset tools:
  * support for really large text files
  * support for byte pair encoding, word-level encoding
