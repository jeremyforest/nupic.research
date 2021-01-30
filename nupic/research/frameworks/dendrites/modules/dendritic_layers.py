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

"""
A simple implementation of dendrite weights. This combines the output from a (sparse)
linear layer with the output from a set of dendritic segments.
"""
import abc

import nupic.research.frameworks.dendrites.functional as dendrite_fxns
from nupic.torch.modules.sparse_weights import SparseWeights, SparseWeights2d

from .dendrite_segments import DendriteSegments


class DendriticLayerBase(SparseWeights, metaclass=abc.ABCMeta):
    """
    Base class for all Dendritic Layer modules.

    This combines a DendriteSegments module with a SparseLinear module.
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        TODO: specify the type - what is module_sparsity type?
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.segments = None
        super().__init__(module, sparsity=module_sparsity)

        self.segments = DendriteSegments(
            num_units=module.weight.shape[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)


class BiasingDendriticLayer(DendriticLayerBase):
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a bias."""
        return dendrite_fxns.dendritic_bias_1d(y, dendrite_activations)


class GatingDendriticLayer(DendriticLayerBase):
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        # Multiple by the sigmoid of the max along each segment.
        return dendrite_fxns.dendritic_gate_1d(y, dendrite_activations)


class AbsoluteMaxGatingDendriticLayer(DendriticLayerBase):
    """
    This layer is similar to `GatingDendriticLayer`, but selects dendrite activations
    based on absolute max activation values instead of just max activation values. For
    example, if choosing between activations -7.4, and 6.5 for a particular unit, -7.4
    will be chosen, and its sign will be kept.
    """

    def apply_dendrites(self, y, dendrite_activations):
        return dendrite_fxns.dendritic_absolute_max_gate_1d(y, dendrite_activations)


class DendriticLayer2dBase(SparseWeights2d, metaclass=abc.ABCMeta):
    """
    Base class for all 2d Dendritic Layer modules.

    Similar to the DendriticLayerBase class, the output from the dendrite segments
    is applied to the output of each channel. Thus, each channel output gets
    modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        :param module: conv2d module which performs the forward pass
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.segments = None
        super().__init__(module, sparsity=module_sparsity)

        self.segments = DendriteSegments(
            num_units=module.out_channels,
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """
        Computes the forward pass through the `torch.nn.Conv2d` module and applies the
        output of the dendrite segments.
        """
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)


class GatingDendriticLayer2d(DendriticLayer2dBase):
    """
    A convolutional version of `GatingDendriticLayer`. The multiplicative dendrite
    outputs are applied element-wise to each output channel. That is, for a given
    output channel, all activation values (determined by the convolution operation) are
    multiplied by a single value computed via dendrites.
    """

    def apply_dendrites(self, y, dendrite_activations):
        """
        Returns the output of the gating convolutional dendritic layer by multiplying
        all values in each output channel by the selected dendrite activations.
        Dendrite activations are selected based on the maximum activations across all
        segments for each channel. Each channel has its own set of dendritic weights,
        and the selected activation is based on the the absolute max value.

        :param y: output of the convolution operation (a torch tensor with shape
                  (b, c, h, w) where the axes represent the batch, channel, height, and
                  width dimensions respectively)
        :param dendrite_activations: the dendrite activation values (a torch tensor
                                     with shape (b, c) where the axes represent the
                                     batch and channel dimensions, respectively)
        """
        return dendrite_fxns.dendritic_gate_2d(y, dendrite_activations)


class AbsoluteMaxGatingDendriticLayer2d(DendriticLayer2dBase):
    """
    A convolutional version of `AbsoluteMaxGatingDendriticLayer`.
    """

    def apply_dendrites(self, y, dendrite_activations):
        """
        Returns the output of the absolute max gating convolutional dendritic layer by
        multiplying all values in each output channel by the selected dendrite
        activations. Dendrite activations are selected based on the absolute maximum
        activations (keeping the sign) across all segments for each channel. Each
        channel has its own set of dendritic weights, and the selected activation is
        based on the the absolute max value.

        :param y: output of the convolution operation (a torch tensor with shape
                  (b, c, h, w) where the axes represent the batch, channel, height, and
                  width dimensions respectively)
        :param dendrite_activations: the dendrite activation values (a torch tensor
                                     with shape (b, c) where the axes represent the
                                     batch and channel dimensions, respectively)
        """
        return dendrite_fxns.dendritic_absolute_max_gate_2d(y, dendrite_activations)
