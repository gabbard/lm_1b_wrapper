#!/usr/bin/env python
"""
Applies the Google Billion word language model to a file of sentences.

The input is expected to be a tab-separated values file with the sentence in the first field as
a sequence of space-separated tokens.

The output will be the same as the input except a language model probability of each sentence will
be inserted as a new first field.  The remaining fields on each row will be the same as the input
fields.

Note that providing a list of sentences, one-per-line, as input is valid.

If the parameter `profile` is set to `True`, profiling information will be dumped to a file in the
working directory with the prefix `profile_`.
"""
import csv
import os
import time
from contextlib import AbstractContextManager

import tensorflow.contrib.tfprof
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from lm1b_wrapper.lm1b import LM1B


class NoOpContextManager(AbstractContextManager):
    def __exit__(self, exc_type, exc_value, traceback):
        pass


def main(params: Parameters):
    graph_def_file = params.existing_file("graph_def_file")
    checkpoint_glob = params.string("checkpoint_glob")
    vocab_file = params.existing_file("vocab_file")
    sentences_file = params.existing_file("sentences_file")
    output_file = params.creatable_file("output_file")
    do_profiling = params.optional_boolean_with_default("profile", False)

    with tensorflow.contrib.tfprof.ProfileContext(
            os.getcwd(),
            trace_steps=range(2, 10),
            dump_steps=range(1, 10, 2),
            enabled=params.optional_boolean_with_default("profile", False)
        ):
        lm = LM1B.load(graph_def_file=graph_def_file,
                       checkpoint_file=checkpoint_glob,
                       vocab=vocab_file)

        start_time = None
        num_tokens_processed = 0

        with open(sentences_file, 'r', newline='') as inp:
            csv_input = csv.reader(inp, delimiter='\t')
            with open(output_file, 'w', newline='') as out:
                csv_output = csv.writer(out, delimiter='\t')
                for line in csv_input:
                    tokens = line[0].split(' ')
                    output_row = list(line)
                    output_row.insert(0, lm.log_probability_of_sentence(tokens))
                    csv_output.writerow(output_row)
                    # we delay till after the first sentence to avoid counting startup time
                    if num_tokens_processed == 0:
                        start_time = time.time()
                    num_tokens_processed += len(tokens)

    elapsed_time = time.time() - start_time
    print(f"Processed {num_tokens_processed - 1} sentences in {elapsed_time} "
          f"seconds, {num_tokens_processed / elapsed_time} tokens per second. First sentence not "
          f"included in time calculation.")


if __name__ == "__main__":
    parameters_only_entry_point(main)
