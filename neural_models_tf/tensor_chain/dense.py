"""
    Author      :   Ermal Rrapaj, code based on KERAS
    Description :   matrix product operator dense layer
                              _     _
                             ( /---\ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

"""Contains the Dense layer."""

import tensorflow as tf

from tensorflow.keras import backend

from anthe_official.neural_models_tf.tensor_chain.utils import get_tc_kernel


class TCDenseOld(tf.keras.layers.Dense):
    def __init__(self, units, tc_length=3, bond=None, ratio=None, **kwargs):
        super().__init__(units, **kwargs)

        self.bond = bond
        self.ratio = ratio
        self.length = int(tc_length)

        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        self.kernel = get_tc_kernel(self, int(last_dim), int(self.units),
                                     length=self.length, bond=self.bond, ratio=self.ratio, name=self._name,
                                     initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def get_config(self):
        config = {
            "bond": self.bond,
            "ratio": self.ratio,
            "length": self.length,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))




class TCDense(tf.keras.layers.Dense):
    def __init__(self, units, tc_length=3, bond=None, ratio=None, **kwargs):
        super().__init__(units, **kwargs)

        self.bond = bond
        self.ratio = ratio
        self.length = int(tc_length)

        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        if self.last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        self.kernels, self.einsum_string = get_tc_kernel(self, int(self.last_dim), int(self.units),
                                     length=self.length, bond=self.bond, ratio=self.ratio, name=self._name,
                                     initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint, return_tensors_and_einsum_string=True)

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def get_config(self):
        config = {
            "bond": self.bond,
            "ratio": self.ratio,
            "length": self.length,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):


        kernel = tf.einsum(self.einsum_string, *self.kernels)
        kernel = tf.reshape(kernel, [self.last_dim, self.units])

        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = tf.matmul(inputs, kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)

if __name__ == '__main__':
    tc = TCDense(10, tc_length=3, bond=None, ratio=0.5)
    tc.build((None, 5, 3))

    names = [weight.name for layer in [tc] for weight in layer.weights]
    weights = tc.get_weights()
    print(names)