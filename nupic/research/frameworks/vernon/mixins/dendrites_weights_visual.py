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
import matplotlib.pyplot as plt

__all__ = ["DendritesWeightsVisual"]


class DendritesWeightsVisual:
    """
    A way to access and graph dendrites weights to follow up sparsity and
    pruning.
    """

    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters
        - dendrite_weights_visual: boolean, if True will generate plot
        """

        super().setup_experiment(config)
        self.dendrite_weights_visual = config.get(
            "dendrite_weights_visual", {False})

    def post_epoch(self):
        total_params_over_epochs = []
        total_nonzero_params_over_epochs = []
        super().post_epoch()
        for module in self.model.modules():
            if 'segments' in module._modules.keys():
                total_params, total_nonzero_params = count_nonzero_params(
                    module.module)
                total_params_over_epochs.append(total_params)
                total_nonzero_params_over_epochs.append(total_nonzero_params)
                self.current_timestep += 1
        if self.dendrite_weights_visual:
            graph_parameters(total_params_over_epochs,
                             total_nonzero_params_over_epochs)

    def graph_parameters(total_params, total_nonzero_params):
        import pdb
        pdb.set_trace()
        zero_fraction = total_nonzero_params / total_params
        plt.plot(zero_fraction, epochs)
        plt.show()
