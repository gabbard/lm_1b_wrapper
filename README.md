# lm_1b_wrapper

Google provides a language model trained on 1 billion words
[here](https://github.com/tensorflow/models/tree/master/research/lm_1b). However, while they provide a script for doing
 a few things with that model from the command-line, there is no nice API for programmatic access.
 This repository provides such a limited version of such an API. At the moment, it only supports getting the
 log-probability of a sentence.

It targets Python 3.6.

# Installation

```
pip install .
```
