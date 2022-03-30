#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with librispeech.
The tokenizer coverts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.

Authors
 * Abdel Heba 2021

Adapted by Ruhr-University Bochum 2021
"""

import sys
import pandas as pd
from os.path import join
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

def join_db(db_list, outfile):
    dfs = [pd.read_json(db, orient='index') for db in db_list]
    df = pd.concat(dfs)
    df.to_csv(outfile)

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # join split files
    # TODO This seems to be complicated. Why not just give the tokenizer train and test files?
    join_db([join(hparams["data_folder"], n) for n in hparams["train_splits"]], hparams["train"])
    join_db([join(hparams["data_folder"], n) for n in hparams["test_splits"]], hparams["test"])
    join_db([join(hparams["data_folder"], n) for n in hparams["dev_splits"]], hparams["dev"])

    # Train tokenizer
    hparams["tokenizer"]()
