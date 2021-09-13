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

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins

"""Permuted MNIST with DendriticMLP"""


class SparseDendriteExperiment(
    mixins.PruneLowMagnitude,
    mixins.DendritesWeightsVisual,
    mixins.RezeroWeights,
    mixins.CentroidContext,
    mixins.PermutedMNISTTaskIndices,
    DendriteContinualLearningExperiment,
):
    pass


TEST = dict(
    experiment_class=SparseDendriteExperiment,
    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites/"),
    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        seed=42,
        download=False,  # Change to True if running for the first time
    ),
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[100, 100],
        num_segments=3,
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.5,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),
    batch_size=256,
    val_batch_size=512,
    epochs=3,
    num_tasks=10,
    tasks_to_validate=range(10),  # Tasks on which to run validate
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=5,
    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=5e-4),
    prune_schedule=[[0, 0.3], [1, 0.6], [2, 1.]],
    prune_curve_shape='exponential',
    validate_on_prune=False,
    dendrite_weights_visual=True,
)

CONFIGS = dict(test=TEST)
