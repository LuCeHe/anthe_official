"""
    Author      :   Ermal Rrapaj, code based on KERAS
    Description :   matrix product operator convolution layer
                              _     _
                             ( /---\ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

"""base class for convolution layers."""

import tensorflow as tf

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.input_spec import InputSpec

from anthe_official.neural_models_tf.tensor_chain.utils import get_tc_kernel
from anthe_official.neural_models_tf.tensor_chain.base_conv import Conv


class TCConv(Conv):

    def __init__(self, rank, filters, kernel_size, tc_length=3, bond=None, ratio=None, **kwargs):
        super().__init__(rank, filters, kernel_size, **kwargs)
        self._bond = bond
        self._ratio = ratio
        self._tc_length = tc_length

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                "number of groups. Received: groups={}, filters={}".format(
                    self.groups, self.filters
                )
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0(s). "
                "Received: %s" % (self.kernel_size,)
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0(s). "
                "Received: %s" % (self.strides,)
            )

        if self.padding == "causal":

            if not isinstance(self, (tf.keras.layers.Conv1D, tf.keras.layers.SeparableConv1D,
                                     TCConv1D)):
                raise ValueError(
                    "Causal padding is only supported for `Conv1D`"
                    "and `SeparableConv1D`."
                )

    def build(self, input_shape):
        
        #get_tc_kernel(self, input_size, output_size, length, bond, ratio, initializer, regularizer, constraint)
        
        
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                "the number of groups. Received groups={}, but the input "
                "has {} channels (full input shape is {}).".format(
                    self.groups, input_channel, input_shape
                )
            )
        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        #self._input_dims = get_three_factors(self.kernel_size[0])
        #self._output_dims = get_three_factors(input_channel // self.groups)

        #if self._bond is None or self._bond < 1:
        #    if self._ratio is None or self._ratio < 0:
        #        r = 0.2
        #    else:
        #        r = self._ratio


        #    self._bond = max(2, int((-self._input_dims[0] * self._output_dims[0]
        #                             - self._input_dims[2] * self._output_dims[2]
        #                             + sqrt((self._input_dims[0] * self._output_dims[0]
        #                                     + self._input_dims[2] * self._output_dims[2]) ** 2
        #                                    + 4 * self._input_dims[1] * self._output_dims[1]
        #                                    * self.kernel_size[0] * (input_channel // self.groups) * r))
        #                            / (2 * self._input_dims[1] * self._output_dims[1])))
        #    self._ratio = self.kernel_size[0] * (input_channel // self.groups) / self._bond / (
        #            self._input_dims[0] * self._output_dims[0] + self._input_dims[2] * self._output_dims[2] +
        #            self._input_dims[1] * self._output_dims[1] * self._bond)

        # compute_output_shape contains some validation logic for the input
        # shape, and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)

        self.kernel = []

        for i in range(self.filters):
            # print(self._name + "_"+str(i))
            var = get_tc_kernel(self, self.kernel_size[0], (input_channel // self.groups), 
                                 self._tc_length, self._bond, self._ratio, self._name + "_"+str(i),
                   self.kernel_initializer, self.kernel_regularizer, self.kernel_constraint)
            #kernels = []
            #kernels.append(self.add_weight("kernel_0%d" % (i),
            #                               shape=[self._input_dims[0], self._bond, self._output_dims[0]],
            #                               initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            #                               constraint=self.kernel_constraint, trainable=True, dtype=self.dtype))
            #kernels.append(self.add_weight("kernel_1%d" % (i),
            #                               shape=[self._input_dims[1], self._bond, self._bond, self._output_dims[1]],
            #                               initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            #                               constraint=self.kernel_constraint, trainable=True, dtype=self.dtype))
            #kernels.append(self.add_weight("kernel_2%d" % (i),
            #                               shape=[self._input_dims[2], self._bond, self._output_dims[2]],
            #                               initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            #                               constraint=self.kernel_constraint, trainable=True, dtype=self.dtype))

            #var = tf.einsum('ijk,ljmr,smp->ilskrp', *kernels)
            #var = tf.reshape(var, [self.kernel_size[0], (input_channel // self.groups)])

            self.kernel.append(var)

        self.kernel = tf.stack(self.kernel, axis=-1)

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        self.built = True

    def get_config(self):
        config = {
            "bond": self._bond,
            "ratio": self._ratio,
            "tc_length": self._tc_length,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TCConv1D(TCConv):
    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            padding="valid",
            data_format="channels_last",
            dilation_rate=1,
            groups=1,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )


# Alias

TCConvolution1D = TCConv1D
