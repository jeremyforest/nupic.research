#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Finetuning on Squad, not Glue.
"""

from copy import deepcopy

from transformers import Trainer

from .finetuning import finetuning_bert_100k_glue_get_info
from callbacks import RezeroWeightsCallback, TrackEvalMetrics
from trainer_mixins import QuestionAnsweringMixin

class QuestionAnsweringTrainer(QuestionAnsweringMixin, Trainer):
    pass


debug_bert_squad = deepcopy(finetuning_bert_100k_glue_get_info)
debug_bert_squad.update(
    # Model Args
    model_name_or_path="bert-base-uncased",
    finetuning=True,
    task_names=None,
    task_name="squad",
    dataset_name="squad",
    dataset_config_name="plain_text",
    trainer_class=QuestionAnsweringTrainer,
    max_seq_length=128,
    do_train=True,
    do_eval=True,
    do_predict=True,
    trainer_callbacks=[
        TrackEvalMetrics(),
    ],
    max_steps=100,
    eval_steps=20,
    rm_checkpoints=True,
    load_best_model_at_end=True,
    warmup_ratio=0.
)

# Supposed to train in about 24 minutes
# Expect f1 score of 88.52, exact_match of 81.22
bert_squad_replication = deepcopy(debug_bert_squad)
bert_squad_replication.update(
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=2,
    max_seq_length=384,
    doc_stride=128,
    learning_rate=3e-5,
    save_steps=1_000,
    eval_steps=1_000,
    logging_steps=100_000, # intended to not log until the end
    # trainer_callbacks=[],
)

del bert_squad_replication["max_steps"]

bert_squad_replication_cased = deepcopy(bert_squad_replication)
bert_squad_replication_cased.update(
    model_name_or_path="bert-base-cased"
)

bert_squad_debug_tracking = deepcopy(bert_squad_replication_cased)
bert_squad_debug_tracking.update(
    save_steps=50,
    eval_steps=50,
    max_steps=500
)

bert_100k_squad = deepcopy(bert_squad_replication_cased)
bert_100k_squad.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501
)

# Run with a different name when testing in env
# with different versions of transformers,datasets, tokenizers
bert_squad_replication_beam_search = deepcopy(bert_squad_replication)
bert_squad_replication_beam_search.update(
    model_name_or_path="xlnet-large-cased",
    beam_search=True,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    save_steps=5_000,
    eval_steps=5_000,
)

bert_squad2_replication = deepcopy(bert_squad_replication_beam_search)
bert_squad2_replication.update(
    dataset_name="squad_v2",
    version_2_with_negative=True,
    num_train_epochs=4,
)

# Export configurations in this file
CONFIGS = dict(
    debug_bert_squad=debug_bert_squad,
    bert_squad_replication=bert_squad_replication,
    bert_squad_replication_beam_search=bert_squad_replication_beam_search,
    bert_squad2_replication=bert_squad2_replication,
    bert_squad_replication_cased=bert_squad_replication_cased,
    bert_squad_debug_tracking=bert_squad_debug_tracking,
    bert_100k_squad=bert_100k_squad
)
