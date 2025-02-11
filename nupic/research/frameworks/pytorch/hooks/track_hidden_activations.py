# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from .base import TrackStatsHookBase


class TrackHiddenActivationsHook(TrackStatsHookBase):
    """
    Forward hook class for tracking hidden activations of modules.
    """

    def __init__(self, name, max_samples_to_track):
        super().__init__(name=name)

        self.num_samples = max_samples_to_track

        # `_activations` keeps all hidden activations in memory, and grows linearly in
        # space with the number of samples
        # self._activations = None
        self._activations = torch.tensor([])

    def get_statistics(self):
        return (self._activations,)

    def __call__(self, module, x, y):
        """
        Forward hook on torch.nn.Module.

        :param module: module
        :param x: input to module, or tuple of inputs to module
        :param y: output of module
        """
        if not self._tracking:
            return

        self._activations = self._activations.to(y.device)
        self._activations = torch.cat((y, self._activations), dim=0)

        # Keep only the last 'num_samples'
        self._activations = self._activations[:self.num_samples, ...]
