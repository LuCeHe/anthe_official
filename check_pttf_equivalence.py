import torch
import tensorflow as tf
from anthe_official.neural_models_pt import EmbeddingLayer as EmbeddingLayerPT
from anthe_official.neural_models_pt.tensor_chain.dense import TCDense as TCDensePT
from anthe_official.neural_models_tf.tensor_chain.dense import TCDense as TCDenseFT
from anthe_official.neural_models_tf import EmbeddingLayer as EmbeddingLayerTF
from anthe_official.neural_models_pt import SoftPOS as SoftPOSPT
from anthe_official.neural_models_tf import SoftPOS as SoftPOSTF
from anthe_official.neural_models_pt import HSoftPOS as HSoftPOSPT
from anthe_official.neural_models_tf import HSoftPOS as HSoftPOSTF
from anthe_official.neural_models_pt import GEGLU as GEGLUPT
from anthe_official.neural_models_tf import GEGLU as GEGLUTF
from anthe_official.neural_models_pt import PositionWiseFeedForwardLayer as PositionWiseFeedForwardLayerPT
from anthe_official.neural_models_tf import PositionWiseFeedForwardLayer as PositionWiseFeedForwardLayerTF

import numpy as np

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)

vocab_size = 100
d_model = 16
max_sequence_len = 1
batch_size = 3
dilation = 2
kernel_size = 3
pt_channel_axis = -1  # 1 or -1
assert pt_channel_axis in [1, -1]
comments = ''

check_embeddings = False
check_softpos = False
check_conv = False
check_hsoftpos = False
check_ffn = False
check_geglu = False
check_tcdense = True

if check_embeddings:
    sequences = np.random.randint(0, vocab_size, (batch_size, max_sequence_len))
    print(sequences)

    # PT and TF
    embedding_layer_pt = EmbeddingLayerPT(vocab_size, d_model)
    embedding_layer_tf = EmbeddingLayerTF(vocab_size, d_model)

    # Initialize weights
    embedding_layer_tf.build((batch_size, max_sequence_len))
    embedding_layer_tf.embedding.build((batch_size, max_sequence_len))
    # emb_matrix_tf = embedding_layer_tf.embedding.embeddings.numpy()
    emb_matrix_tf = embedding_layer_pt.embedding.weight.detach().numpy()

    # emb_matrix_tf = emb_matrix_pt
    embedding_layer_tf.embedding.embeddings.assign(emb_matrix_tf)

    # Run
    output_pt = embedding_layer_pt(torch.from_numpy(sequences))
    output_tf = embedding_layer_tf(sequences)

    # Compare
    print('Are the Embedding TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_softpos:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    softpos_layer_pt = SoftPOSPT(d_model, extend_axis=pt_channel_axis)
    softpos_layer_tf = SoftPOSTF(d_model)

    # get weights from PT
    spos = []
    spos_matrices_pt = softpos_layer_pt.spos

    # use it in TF
    softpos_layer_tf.build((batch_size, max_sequence_len, d_model))
    for i in range(len(spos_matrices_pt)):
        spos_pt = softpos_layer_pt.spos[i].detach().numpy()
        softpos_layer_tf.spos[i].assign(spos_pt)

    # Initialize weights
    output_tf = softpos_layer_tf(input_tensor)

    if pt_channel_axis == -1:
        output_pt = softpos_layer_pt(torch.from_numpy(input_tensor))
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = softpos_layer_pt(input_tensor)
        output_pt = torch.transpose(output_pt, 1, 2)

    print('Are the SoftPOS TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_conv:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    conv1d_pt = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model // 2, kernel_size=kernel_size, stride=1,
                                padding=0,
                                dilation=dilation)
    conv1d_tf = tf.keras.layers.Conv1D(filters=d_model // 2, kernel_size=kernel_size, strides=1, padding='causal',
                                       dilation_rate=dilation)
    output_tf = conv1d_tf(input_tensor)

    conv_weight_pt = conv1d_pt.weight.detach().numpy()
    conv_bias_pt = conv1d_pt.bias.detach().numpy()
    conv_weight_pt = np.moveaxis(conv_weight_pt, 0, -1)
    conv_weight_pt = np.moveaxis(conv_weight_pt, 0, 1)

    conv1d_tf.build((batch_size, max_sequence_len, d_model))

    conv1d_tf.kernel.assign(conv_weight_pt)
    conv1d_tf.bias.assign(conv_bias_pt)

    output_tf = conv1d_tf(input_tensor)

    input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
    input_tensor = torch.nn.functional.pad(input_tensor, ((kernel_size - 1) * dilation, 0, 0, 0))

    output_pt = conv1d_pt(input_tensor)
    output_pt = torch.transpose(output_pt, 1, 2)
    print('Are the Conv1D TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_hsoftpos:
    n_layers = 2

    sequences = np.random.randint(0, vocab_size, (batch_size, max_sequence_len))

    hsoftpos_layer_pt = HSoftPOSPT(vocab_size, d_model, n_layers=n_layers, extend_axis=pt_channel_axis)
    hsoftpos_layer_tf = HSoftPOSTF(vocab_size, d_model, n_layers=n_layers)

    # Initialize TF embeddings as PT
    hsoftpos_layer_tf.emb.build((batch_size, max_sequence_len))
    hsoftpos_layer_tf.emb.embedding.build((batch_size, max_sequence_len))
    emb_matrix_pt = hsoftpos_layer_pt.emb.embedding.weight.detach().numpy()
    hsoftpos_layer_tf.emb.embedding.embeddings.assign(emb_matrix_pt)

    # print('Are TF and PT embs equivalent?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

    # Initialize TF SoftPOS as PT
    for j in range(len(hsoftpos_layer_pt.spos)):
        spos_matrices_pt = hsoftpos_layer_pt.spos[j].spos

        hsoftpos_layer_tf.spos[j].build((batch_size, max_sequence_len, d_model))
        for i in range(len(spos_matrices_pt)):
            spos_pt = hsoftpos_layer_pt.spos[j].spos[i].detach().numpy()
            hsoftpos_layer_tf.spos[j].spos[i].assign(spos_pt)

    local_d = int(d_model / 2 / n_layers)
    embd_d = d_model - local_d * (2 * n_layers - 1)
    # Initialize TF Convs as PT
    for j in range(len(hsoftpos_layer_pt.convs)):
        conv_weight_pt = hsoftpos_layer_pt.convs[j].weight.detach().numpy()
        conv_bias_pt = hsoftpos_layer_pt.convs[j].bias.detach().numpy()
        conv_weight_pt = np.moveaxis(conv_weight_pt, 0, -1)
        conv_weight_pt = np.moveaxis(conv_weight_pt, 0, 1)

        conv_d_in = embd_d if j == 0 else local_d
        hsoftpos_layer_tf.convs[j].build((batch_size, max_sequence_len, conv_d_in))

        hsoftpos_layer_tf.convs[j].kernel.assign(conv_weight_pt)
        hsoftpos_layer_tf.convs[j].bias.assign(conv_bias_pt)

        conv_weight_tf = hsoftpos_layer_tf.convs[j].kernel.numpy()
        conv_bias_tf = hsoftpos_layer_tf.convs[j].bias.numpy()

    # Initialize weights
    output_tf = hsoftpos_layer_tf(sequences)
    if pt_channel_axis == -1:
        output_pt = hsoftpos_layer_pt(torch.from_numpy(sequences))
    else:
        output_pt = hsoftpos_layer_pt(torch.from_numpy(sequences))
        output_pt = torch.transpose(output_pt, 1, 2)

    print('Are the HSoftPOS TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_ffn:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')

    ffn_pt = PositionWiseFeedForwardLayerPT(4 * d_model, d_model, comments)
    ffn_tf = PositionWiseFeedForwardLayerTF(4 * d_model, d_model, comments)

    w_1 = ffn_pt.w_1.weight.detach().numpy().T
    b_1 = ffn_pt.w_1.bias.detach().numpy().T
    w_2 = ffn_pt.w_2.weight.detach().numpy().T
    b_2 = ffn_pt.w_2.bias.detach().numpy().T

    ffn_tf.build((batch_size, max_sequence_len, d_model))
    ffn_tf.w_1.build((batch_size, max_sequence_len, d_model))
    ffn_tf.w_2.build((batch_size, max_sequence_len, 4 * d_model))

    ffn_tf.w_1.kernel.assign(w_1)
    ffn_tf.w_1.bias.assign(b_1)
    ffn_tf.w_2.kernel.assign(w_2)
    ffn_tf.w_2.bias.assign(b_2)

    # Initialize weights
    output_tf = ffn_tf(input_tensor)

    if pt_channel_axis == -1:
        output_pt = ffn_pt(torch.from_numpy(input_tensor))
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = ffn_pt(input_tensor)
        output_pt = torch.transpose(output_pt, 1, 2)

    print('Are the FFN TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))
    print('Only works for pt_channel_axis == -1')

if check_geglu:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    geglu_layer_pt = GEGLUPT(4 * d_model, d_model, comments='')
    geglu_layer_tf = GEGLUTF(4 * d_model, d_model, comments='')

    w1_pt = geglu_layer_pt.w_1.weight.detach().numpy().T
    w2_pt = geglu_layer_pt.w_2.weight.detach().numpy().T
    w3_pt = geglu_layer_pt.w_3.weight.detach().numpy().T

    b1_pt = geglu_layer_pt.w_1.bias.detach().numpy().T
    b2_pt = geglu_layer_pt.w_2.bias.detach().numpy().T
    b3_pt = geglu_layer_pt.w_3.bias.detach().numpy().T

    geglu_layer_tf.build((batch_size, max_sequence_len, d_model))
    geglu_layer_tf.w_1.build((batch_size, max_sequence_len, d_model))
    geglu_layer_tf.w_2.build((batch_size, max_sequence_len, 2 * 4 * d_model // 3))
    geglu_layer_tf.w_3.build((batch_size, max_sequence_len, d_model))

    geglu_layer_tf.w_1.kernel.assign(w1_pt)
    geglu_layer_tf.w_1.bias.assign(b1_pt)
    geglu_layer_tf.w_2.kernel.assign(w2_pt)
    geglu_layer_tf.w_2.bias.assign(b2_pt)
    geglu_layer_tf.w_3.kernel.assign(w3_pt)
    geglu_layer_tf.w_3.bias.assign(b3_pt)

    # Initialize weights
    output_tf = geglu_layer_tf(input_tensor)

    if pt_channel_axis == -1:
        output_pt = geglu_layer_pt(torch.from_numpy(input_tensor))
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = geglu_layer_pt(input_tensor)
        output_pt = torch.transpose(output_pt, 1, 2)

    print('Are the GEGLU TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_tcdense:
    tc_length = 3
    ratio = .1
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    tcdense_pt = TCDensePT(d_model, d_model, tc_length=tc_length, ratio=ratio)
    tcdense_tf = TCDenseFT(d_model, tc_length=tc_length, ratio=ratio)

    w_1 = tcdense_pt.weight.detach().numpy()
    b_1 = tcdense_pt.bias.detach().numpy().T

    tcdense_tf.build((batch_size, max_sequence_len, d_model))

    # tcdense_tf.kernel.assign(w_1)
    # tcdense_tf.bias.assign(b_1)
    tcdense_tf.kernel = w_1
    tcdense_tf.bias = b_1

    output_tf = tcdense_tf(input_tensor)

    if pt_channel_axis == -1:
        output_pt = tcdense_pt(torch.from_numpy(input_tensor))
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = tcdense_pt(input_tensor)
        output_pt = torch.transpose(output_pt, 1, 2)

    print(output_pt)
    print(output_tf)

    print('Are the TCDense TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))
