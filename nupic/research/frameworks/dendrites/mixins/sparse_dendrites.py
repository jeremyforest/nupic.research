# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


import torch
import torch.nn.functional as F
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params

__all__ = ["SparseDendritesPlasticity"]


class SparseDendritesPlasticity:
    """
    Implement plasticity on sparse dendrite segments.

    1. Get dendrites weights values
    2. Calculate threshold to stay at defined sparsity
    3. If weight < X% threshold --> reset to 0 , else do nothing. This X %
    can be adjusted and is dependent on how many new weights are initialized.
    The more new weights, the more pruning and the higher this threshold. As an
    example, having 50% new weights means we have set the threshold a 200%.
    4. Set zero weights to non-zero randomly. Match defined sparsity using this
    number and the step 3 threshold.
    5. Iterate every X epochs



    Example config:
    ```
    config=dict(
        sparse_dendrites=dict(
            plasticity_update=1,
            percent_new_weights=50,
        )
    )
    """

    def setup_experiment(self, config):

        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters
        - plasticity_update: number of epochs between weights pruning/growing update.
        - percent_new_weights: percentage of zero weights updated to non-zero
                               values during plasticity update
        """
        super().setup_experiment(config)
        sparse_dendrites = config.get("sparse_dendrites", {})
        self.plasticity_update = sparse_dendrites.get("plasticity_update", 1)
        self.percent_new_weights = sparse_dendrites.get("percent_new_weights", 50)

        # TODO is this the best way to access the config file parameters ?
        model_args = config.get("model_args")
        self.sparsity = model_args.get("dendrite_weight_sparsity")
        dataset_args = config.get("dataset_args")
        self.epochs = dataset_args.get("epochs")

    def run_task(self):
        ret = super().run_task()
        epochs_to_update = torch.linspace(0, self.epochs, self.plasticity_update)
        if self.epoch in epochs_to_update:
            self.weights = self.model.parameters()
            self.prune_weights(self.weights, self.sparsity)
            self.grow_weights(self.weights, self.percent_new_weights)
        return ret

    def prune_weights(self, weights, sparsity_level):
        # TODO create mask to zero the weights that are X (200% here) lower than threshold
        # TODO the threshold needs to be dynamically set somewhere
        # TODO apply mask
        # TODO return weights
        pass

    def grow_weights(self, weights, percent_new_weights):
        # TODO choose new weights to update respecting percent_new_weights and sparsity parameters
        # TODO update to non-zeros using the standard initialization schema
        pass

    def init_sparse_weights(m, input_sparsity):
        """
        Modified Kaiming weight initialization that considers input sparsity and weight
        sparsity.
        """
        input_density = 1.0 - input_sparsity
        weight_density = 1.0 - m.sparsity
        _, fan_in = m.module.weight.size()
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.module.weight, -bound, bound)
        m.apply(rezero_weights)
