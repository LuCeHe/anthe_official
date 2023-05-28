# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Embedding layer."""

import tensorflow as tf

from anthe_official.neural_models_tf.tensor_chain.utils import get_tc_kernel


class TCEmbedding(tf.keras.layers.Embedding):
    def __init__(self, input_dim, output_dim, tc_length=3, bond=None, ratio=None, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        self._input_size = int(input_dim)
        self._output_size = int(output_dim)
        self._length = int(tc_length)
        self._bond = bond
        self._ratio = ratio

    def build(self, input_shape=None):
        self.embeddings = get_tc_kernel(self, self._input_size, self._output_size,
                                         length=self._length, bond=self._bond, ratio=self._ratio, name=self._name,
                                         initializer=self.embeddings_initializer,
                                         regularizer=self.embeddings_regularizer,
                                         constraint=self.embeddings_constraint)

        self.built = True

    def get_config(self):
        config = {
            "bond": self._bond,
            "ratio": self._ratio,
            "length": self._length,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
