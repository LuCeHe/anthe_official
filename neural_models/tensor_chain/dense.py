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

from anthe_official.neural_models.tensor_chain.utils import get_tc_kernel


class TCDense(tf.keras.layers.Dense):
    def __init__(self, units, tc_length=3, bond=None, ratio=None, **kwargs):
        super().__init__(units, **kwargs)

        self._bond = bond
        self._ratio = ratio
        self._length = int(tc_length)

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
                                     length=self._length, bond=self._bond, ratio=self._ratio, name=self._name,
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
            "bond": self._bond,
            "ratio": self._ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
