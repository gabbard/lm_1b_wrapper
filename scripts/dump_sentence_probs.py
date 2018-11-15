#!/usr/bin/env python
import csv
import time

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from lm1b_wrapper.lm1b import LM1B

def main(params: Parameters):
    graph_def_file = params.existing_file("graph_def_file")
    checkpoint_glob = params.string("checkpoint_glob")
    vocab_file = params.existing_file("vocab_file")
    sentences_file = params.existing_file("sentences_file")
    output_file = params.creatable_file("output_file")

    lm = LM1B.load(graph_def_file=graph_def_file,
                   checkpoint_file=checkpoint_glob,
                   vocab=vocab_file)

    start_time = None
    num_sentences_processed = 0

    with open(sentences_file, 'r') as inp:
        with open(output_file, 'w', newline='') as out:
            csvwriter = csv.writer(out)
            for line in inp:
                if line:
                    tokens = line.split(' ')
                    csvwriter.writerow((tokens, lm.log_probability_of_sentence(tokens)))
                    num_sentences_processed += 1
                    # we delay till after the first sentence to avoid counting startup time
                    if num_sentences_processed == 1:
                        start_time = time.time()

    print(f"Processed {num_sentences_processed - 1} sentences in {time.time() - start_time} "
          f"seconds. First sentence not included in time calculation")

if __name__ == "__main__":
    parameters_only_entry_point(main)
