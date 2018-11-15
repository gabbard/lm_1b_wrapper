import logging
import math
from pathlib import Path
from typing import Any, Mapping, Union, Sequence

import numpy as np
import tensorflow as tf
from attr import attrs, attrib
from google.protobuf import text_format

from data_utils import CharsVocabulary

TensorflowSession = Any
TensorflowNode = Any

# constants taken from lm_1b_eval.py
_BATCH_SIZE = 1
_NUM_TIMESTEPS = 1
_MAX_WORD_LEN = 50
_START_SENTENCE_SYMBOL = "<S>"


@attrs(frozen=True, auto_attribs=True)
class LM1B:
    _session: TensorflowSession
    _name_to_node: Mapping[str, TensorflowNode]
    _vocab: CharsVocabulary

    @staticmethod
    def load(*, graph_def_file: Path, checkpoint_file: Path,
             vocab: Union[Path, CharsVocabulary]) -> 'LM1B':
        resolved_vocab: CharsVocabulary
        if isinstance(vocab, CharsVocabulary):
            resolved_vocab = vocab
        else:
            resolved_vocab = CharsVocabulary(str(vocab), _MAX_WORD_LEN)

        ### copied from Tensorflow's model repo's lm_1b_eval.py
        with tf.Graph().as_default():
            logging.info('Recovering graph.')
            with tf.gfile.FastGFile(str(graph_def_file), 'r') as f:
                s = f.read()
                gd = tf.GraphDef()
                text_format.Merge(s, gd)

            tf.logging.info('Recovering Graph %s', graph_def_file)
            t = {}
            [t['states_init'], t['lstm/lstm_0/control_dependency'],
             t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
             t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
             t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
             t['all_embs'], t['softmax_weights'], t['global_step']
             ] = tf.import_graph_def(gd, {}, ['states_init',
                                              'lstm/lstm_0/control_dependency:0',
                                              'lstm/lstm_1/control_dependency:0',
                                              'softmax_out:0',
                                              'class_ids_out:0',
                                              'class_weights_out:0',
                                              'log_perplexity_out:0',
                                              'inputs_in:0',
                                              'targets_in:0',
                                              'target_weights_in:0',
                                              'char_inputs_in:0',
                                              'all_embs_out:0',
                                              'Reshape_3:0',
                                              'global_step:0'], name='')

            logging.info('Recovering checkpoint %s\n', checkpoint_file)
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run('save/restore_all', {'save/Const:0': str(checkpoint_file)})
            sess.run(t['states_init'])

        return LM1B(session=sess, name_to_node=t, vocab=resolved_vocab)

    def log_probability_of_sentence(self, tokens: Sequence[str]):
        """
        Derived from _SampleModel in lm_1b_eval.py
        """
        if isinstance(tokens, str):
            raise ValueError("Input to log_probability_of_sentence is a sequence of token strings,"
                             " not a single string")
        # these don't matter when we are running the model in inference mode
        targets = np.zeros([_BATCH_SIZE, _NUM_TIMESTEPS], np.int32)
        weights = np.ones([_BATCH_SIZE, _NUM_TIMESTEPS], np.float32)

        # these contain information about the previous word
        # we initialize them with the beginning-of-sentence marker
        inputs = np.zeros([_BATCH_SIZE, _NUM_TIMESTEPS], np.int32)
        inputs[0, 0] = self._vocab.word_to_id(_START_SENTENCE_SYMBOL)

        char_ids_inputs = np.zeros(
            [_BATCH_SIZE, _NUM_TIMESTEPS, self._vocab.max_word_length], np.int32)
        char_ids_inputs[0, 0, :] = self._vocab.word_to_char_ids(_START_SENTENCE_SYMBOL)

        # we take the log probability of a token sequence to be the sum of the log-probs
        # of each of its tokens given the preceding context
        log_prob_sum = 0.0
        for token in tokens:
            dist_over_next_words = self._session.run(
                self._name_to_node['softmax_out'],
                feed_dict={
                    self._name_to_node['char_inputs_in']: char_ids_inputs,
                    self._name_to_node['inputs_in']: inputs,
                    self._name_to_node['targets_in']: targets,
                    self._name_to_node['target_weights_in']: weights})
            token_idx = self._vocab.word_to_id(token)
            log_prob_sum += math.log(dist_over_next_words[0][token_idx])

            # prepare this word to be the context for the next word
            inputs[0, 0] = token_idx
            char_ids_inputs[0, 0, :] = self._vocab.word_to_char_ids(token)

        # restore original state so that future calls to log_probability_of_sentence
        # are not affected by past calls
        self._reset_state()

        return log_prob_sum

    def _reset_state(self) -> None:
        self._session.run(self._name_to_node['states_init'])
