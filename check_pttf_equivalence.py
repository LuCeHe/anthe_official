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
from anthe_official.neural_models_pt import MultiHeadAttention as MultiHeadAttentionPT
from anthe_official.neural_models_tf import MultiHeadAttention as MultiHeadAttentionTF
from anthe_official.neural_models_pt import AntheEncoderBlock as AntheEncoderBlockPT
from anthe_official.neural_models_tf import AntheEncoderBlock as AntheEncoderBlockTF
from anthe_official.neural_models_pt import EncoderLayer as EncoderLayerPT
from anthe_official.neural_models_tf import EncoderLayer as EncoderLayerTF
from anthe_official.neural_models_pt import AntheDecoderBlock as AntheDecoderBlockPT
from anthe_official.neural_models_tf import AntheDecoderBlock as AntheDecoderBlockTF

import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)

vocab_size = 100
d_model = 8
max_sequence_len = 3
batch_size = 2
dilation = 2
kernel_size = 3
pt_channel_axis = -1  # 1 or -1
assert pt_channel_axis in [1, -1]
comments = ''

print('Remember that you still have to check that they match with the masking for both axis')
check_embeddings = True
check_softpos = True
check_conv = True
check_ln = True
check_hsoftpos = True
check_ffn = True
check_geglu = True
check_tcdense = True
check_mha = True
check_enclayer = False
check_antheenc = True
check_anthedec = True

if check_embeddings:
    sequences = np.random.randint(0, vocab_size, (batch_size, max_sequence_len))

    # PT and TF
    embedding_layer_pt = EmbeddingLayerPT(vocab_size, d_model, axis=pt_channel_axis)
    embedding_layer_tf = EmbeddingLayerTF(vocab_size, d_model)

    # Initialize weights
    embedding_layer_tf.build((batch_size, max_sequence_len))
    embedding_layer_tf.embedding.build((batch_size, max_sequence_len))
    # emb_matrix_tf = embedding_layer_tf.embedding.embeddings.numpy()
    emb_matrix_tf = embedding_layer_pt.embedding.weight.detach().numpy()

    # emb_matrix_tf = emb_matrix_pt
    embedding_layer_tf.embedding.embeddings.assign(emb_matrix_tf)

    # Run
    output_tf = embedding_layer_tf(sequences)

    if pt_channel_axis == -1:
        output_pt = embedding_layer_pt(torch.from_numpy(sequences))
    else:
        input_tensor = torch.from_numpy(sequences)
        output_pt = embedding_layer_pt(input_tensor)
        output_pt = torch.transpose(output_pt, 1, 2)

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
                                padding=0, dilation=dilation)
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

if check_ln:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    ln_pt = torch.nn.LayerNorm(d_model, eps=1e-6)
    ln_tf = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    ln_weight_pt = ln_pt.weight.detach().numpy()
    ln_bias_pt = ln_pt.bias.detach().numpy()

    ln_tf.build((batch_size, max_sequence_len, d_model))

    ln_tf.gamma.assign(ln_weight_pt)
    ln_tf.beta.assign(ln_bias_pt)

    output_pt = ln_pt(torch.from_numpy(input_tensor))
    output_tf = ln_tf(input_tensor)
    # print(output_pt)
    # print(output_tf)

    print('Are the LayerNorm TF == PT?',
          np.allclose(output_pt.detach().numpy(), output_tf.numpy(), rtol=1.e-4, atol=1.e-7))

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

    ffn_pt = PositionWiseFeedForwardLayerPT(4 * d_model, d_model, comments, axis=pt_channel_axis)
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

if check_geglu:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    geglu_layer_pt = GEGLUPT(4 * d_model, d_model, comments='', axis=pt_channel_axis)
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
    tcdense_pt = TCDensePT(d_model, d_model, tc_length=tc_length, ratio=ratio, axis=pt_channel_axis)
    tcdense_tf = TCDenseFT(d_model, tc_length=tc_length, ratio=ratio)

    w_1 = tcdense_pt.weight.detach().numpy()
    b_1 = tcdense_pt.bias.detach().numpy().T

    tcdense_tf.build((batch_size, max_sequence_len, d_model))

    tcdense_tf.kernel = w_1
    tcdense_tf.bias = b_1

    output_tf = tcdense_tf(input_tensor)

    if pt_channel_axis == -1:
        output_pt = tcdense_pt(torch.from_numpy(input_tensor))
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = tcdense_pt(input_tensor)
        output_pt = torch.transpose(output_pt, 1, 2)

    # print(output_pt)
    # print(output_tf)

    print('Are the TCDense TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_mha:
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')

    att_pt = MultiHeadAttentionPT(4, d_model, comments, axis=pt_channel_axis)
    att_tf = MultiHeadAttentionTF(4, d_model, comments)

    w_q = att_pt.w_query.weight.detach().numpy().T
    b_q = att_pt.w_query.bias.detach().numpy().T
    w_k = att_pt.w_key.weight.detach().numpy().T
    b_k = att_pt.w_key.bias.detach().numpy().T
    w_v = att_pt.w_value.weight.detach().numpy().T
    b_v = att_pt.w_value.bias.detach().numpy().T

    w_ff = att_pt.ff.weight.detach().numpy().T
    b_ff = att_pt.ff.bias.detach().numpy().T

    att_tf.w_query.build((batch_size, max_sequence_len, d_model))
    att_tf.w_key.build((batch_size, max_sequence_len, d_model))
    att_tf.w_value.build((batch_size, max_sequence_len, d_model))
    att_tf.ff.build((batch_size, max_sequence_len, d_model))

    att_tf.w_query.kernel.assign(w_q)
    att_tf.w_query.bias.assign(b_q)
    att_tf.w_key.kernel.assign(w_k)
    att_tf.w_key.bias.assign(b_k)
    att_tf.w_value.kernel.assign(w_v)
    att_tf.w_value.bias.assign(b_v)
    att_tf.ff.kernel.assign(w_ff)
    att_tf.ff.bias.assign(b_ff)

    # Initialize weights
    output_tf = att_tf([input_tensor, input_tensor, input_tensor, None])[0]

    if pt_channel_axis == -1:
        x = torch.from_numpy(input_tensor)
        output_pt = att_pt(x, x, x, None)[0]
    else:
        x = torch.from_numpy(input_tensor)
        x = torch.transpose(x, 1, 2)
        output_pt = att_pt(x, x, x, None)[0]
        output_pt = torch.transpose(output_pt, 1, 2)

    print('Are the MHA TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy()))

if check_enclayer:
    tc_length = 3
    ratio = .1
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    comments = 'geglu_gateattention'  # geglu
    # comments = ''  # geglu
    enclayer_pt = EncoderLayerPT(4, d_model, 4 * d_model, 0.0, comments)
    enclayer_tf = EncoderLayerTF(4, d_model, 4 * d_model, 0.0, comments)

    # LN 1
    ln_weight_pt = enclayer_pt.layer_norm_1.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_1.bias.detach().numpy()

    enclayer_tf.layer_norm_1.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_1.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_1.beta.assign(ln_bias_pt)

    # LN 2
    ln_weight_pt = enclayer_pt.layer_norm_2.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_2.bias.detach().numpy()

    enclayer_tf.layer_norm_2.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_2.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_2.beta.assign(ln_bias_pt)

    # MHA
    w_q = enclayer_pt.attention.w_query.weight.detach().numpy().T
    b_q = enclayer_pt.attention.w_query.bias.detach().numpy().T
    w_k = enclayer_pt.attention.w_key.weight.detach().numpy().T
    b_k = enclayer_pt.attention.w_key.bias.detach().numpy().T
    w_v = enclayer_pt.attention.w_value.weight.detach().numpy().T
    b_v = enclayer_pt.attention.w_value.bias.detach().numpy().T

    w_ff = enclayer_pt.attention.ff.weight.detach().numpy().T
    b_ff = enclayer_pt.attention.ff.bias.detach().numpy().T

    enclayer_tf.attention.w_query.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.w_key.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.w_value.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.ff.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.attention.w_query.kernel = w_q
    enclayer_tf.attention.w_query.bias = b_q
    enclayer_tf.attention.w_key.kernel = w_k
    enclayer_tf.attention.w_key.bias = b_k
    enclayer_tf.attention.w_value.kernel = w_v
    enclayer_tf.attention.w_value.bias = b_v
    enclayer_tf.attention.ff.kernel = w_ff
    enclayer_tf.attention.ff.bias = b_ff

    # GEGLU
    w1_pt = enclayer_pt.position_wise_feed_forward_layer.w_1.weight.detach().numpy().T
    w2_pt = enclayer_pt.position_wise_feed_forward_layer.w_2.weight.detach().numpy().T

    b1_pt = enclayer_pt.position_wise_feed_forward_layer.w_1.bias.detach().numpy().T
    b2_pt = enclayer_pt.position_wise_feed_forward_layer.w_2.bias.detach().numpy().T

    enclayer_tf.position_wise_feed_forward_layer.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.position_wise_feed_forward_layer.w_1.build((batch_size, max_sequence_len, d_model))
    dff = 2 * 4 * d_model // 3 if 'geglu' in comments else 4 * d_model
    enclayer_tf.position_wise_feed_forward_layer.w_2.build((batch_size, max_sequence_len, dff))

    enclayer_tf.position_wise_feed_forward_layer.w_1.kernel = w1_pt
    enclayer_tf.position_wise_feed_forward_layer.w_1.bias = b1_pt
    enclayer_tf.position_wise_feed_forward_layer.w_2.kernel = w2_pt
    enclayer_tf.position_wise_feed_forward_layer.w_2.bias = b2_pt

    if 'geglu' in comments:
        w3_pt = enclayer_pt.position_wise_feed_forward_layer.w_3.weight.detach().numpy().T
        b3_pt = enclayer_pt.position_wise_feed_forward_layer.w_3.bias.detach().numpy().T
        enclayer_tf.position_wise_feed_forward_layer.w_3.build((batch_size, max_sequence_len, d_model))
        enclayer_tf.position_wise_feed_forward_layer.w_3.kernel = w3_pt
        enclayer_tf.position_wise_feed_forward_layer.w_3.bias = b3_pt

    output_tf = enclayer_tf(input_tensor, None)[0]

    if pt_channel_axis == -1:
        output_pt = enclayer_pt(torch.from_numpy(input_tensor), None)[0]
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = enclayer_pt(input_tensor, None)[0]
        output_pt = torch.transpose(output_pt, 1, 2)

    # print(output_pt)
    # print(output_tf)

    print('Are the TransformerEncoder TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy(),
                                                              rtol=1.e-4, atol=1.e-7))

if check_antheenc:
    tc_length = 3
    ratio = .1
    input_tensor = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    enclayer_pt = AntheEncoderBlockPT(4, d_model, 4 * d_model, 0.0, axis=pt_channel_axis)
    enclayer_tf = AntheEncoderBlockTF(4, d_model, 4 * d_model, 0.0)

    # LN 1
    ln_weight_pt = enclayer_pt.layer_norm_1.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_1.bias.detach().numpy()

    enclayer_tf.layer_norm_1.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_1.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_1.beta.assign(ln_bias_pt)

    # LN 2
    ln_weight_pt = enclayer_pt.layer_norm_2.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_2.bias.detach().numpy()

    enclayer_tf.layer_norm_2.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_2.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_2.beta.assign(ln_bias_pt)

    # MHA
    w_q = enclayer_pt.attention.w_query.weight.detach().numpy()
    b_q = enclayer_pt.attention.w_query.bias.detach().numpy().T
    w_k = enclayer_pt.attention.w_key.weight.detach().numpy()
    b_k = enclayer_pt.attention.w_key.bias.detach().numpy().T
    w_v = enclayer_pt.attention.w_value.weight.detach().numpy()
    b_v = enclayer_pt.attention.w_value.bias.detach().numpy().T

    w_ff = enclayer_pt.attention.ff.weight.detach().numpy().T
    b_ff = enclayer_pt.attention.ff.bias.detach().numpy().T

    enclayer_tf.attention.w_query.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.w_key.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.w_value.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.ff.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.attention.w_query.kernel = w_q
    enclayer_tf.attention.w_query.bias = b_q
    enclayer_tf.attention.w_key.kernel = w_k
    enclayer_tf.attention.w_key.bias = b_k
    enclayer_tf.attention.w_value.kernel = w_v
    enclayer_tf.attention.w_value.bias = b_v
    enclayer_tf.attention.ff.kernel = w_ff
    enclayer_tf.attention.ff.bias = b_ff

    # GEGLU
    w1_pt = enclayer_pt.position_wise_feed_forward_layer.w_1.weight.detach().numpy()
    w2_pt = enclayer_pt.position_wise_feed_forward_layer.w_2.weight.detach().numpy()

    b1_pt = enclayer_pt.position_wise_feed_forward_layer.w_1.bias.detach().numpy().T
    b2_pt = enclayer_pt.position_wise_feed_forward_layer.w_2.bias.detach().numpy().T

    enclayer_tf.position_wise_feed_forward_layer.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.position_wise_feed_forward_layer.w_1.build((batch_size, max_sequence_len, d_model))
    dff = 2 * 4 * d_model // 3
    enclayer_tf.position_wise_feed_forward_layer.w_2.build((batch_size, max_sequence_len, dff))

    enclayer_tf.position_wise_feed_forward_layer.w_1.kernel = w1_pt
    enclayer_tf.position_wise_feed_forward_layer.w_1.bias = b1_pt
    enclayer_tf.position_wise_feed_forward_layer.w_2.kernel = w2_pt
    enclayer_tf.position_wise_feed_forward_layer.w_2.bias = b2_pt

    w3_pt = enclayer_pt.position_wise_feed_forward_layer.w_3.weight.detach().numpy()
    b3_pt = enclayer_pt.position_wise_feed_forward_layer.w_3.bias.detach().numpy().T
    enclayer_tf.position_wise_feed_forward_layer.w_3.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.position_wise_feed_forward_layer.w_3.kernel = w3_pt
    enclayer_tf.position_wise_feed_forward_layer.w_3.bias = b3_pt

    output_tf = enclayer_tf(input_tensor, None)[0]

    if pt_channel_axis == -1:
        output_pt = enclayer_pt(torch.from_numpy(input_tensor), None)[0]
    else:
        input_tensor = torch.transpose(torch.from_numpy(input_tensor), 1, 2)
        output_pt = enclayer_pt(input_tensor, None)[0]
        output_pt = torch.transpose(output_pt, 1, 2)

    # print(output_pt)
    # print(output_tf)

    print('Are the AntheEncoder TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy(),
                                                        rtol=1.e-4, atol=1.e-7))

if check_anthedec:
    tc_length = 3
    ratio = .1
    input_tensor_1 = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    input_tensor_2 = np.random.rand(batch_size, max_sequence_len, d_model).astype('float32')
    enclayer_pt = AntheDecoderBlockPT(4, d_model, 4 * d_model, 0.0, axis=pt_channel_axis)
    enclayer_tf = AntheDecoderBlockTF(4, d_model, 4 * d_model, 0.0)

    # LN 1
    ln_weight_pt = enclayer_pt.layer_norm_1.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_1.bias.detach().numpy()

    enclayer_tf.layer_norm_1.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_1.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_1.beta.assign(ln_bias_pt)

    # LN 2
    ln_weight_pt = enclayer_pt.layer_norm_2.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_2.bias.detach().numpy()

    enclayer_tf.layer_norm_2.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_2.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_2.beta.assign(ln_bias_pt)

    # LN 3
    ln_weight_pt = enclayer_pt.layer_norm_3.weight.detach().numpy()
    ln_bias_pt = enclayer_pt.layer_norm_3.bias.detach().numpy()

    enclayer_tf.layer_norm_3.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.layer_norm_3.gamma.assign(ln_weight_pt)
    enclayer_tf.layer_norm_3.beta.assign(ln_bias_pt)

    # MHA
    w_q = enclayer_pt.attention.w_query.weight.detach().numpy()
    b_q = enclayer_pt.attention.w_query.bias.detach().numpy().T
    w_k = enclayer_pt.attention.w_key.weight.detach().numpy()
    b_k = enclayer_pt.attention.w_key.bias.detach().numpy().T
    w_v = enclayer_pt.attention.w_value.weight.detach().numpy()
    b_v = enclayer_pt.attention.w_value.bias.detach().numpy().T

    w_ff = enclayer_pt.attention.ff.weight.detach().numpy().T
    b_ff = enclayer_pt.attention.ff.bias.detach().numpy().T

    enclayer_tf.attention.w_query.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.w_key.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.w_value.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.attention.ff.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.attention.w_query.kernel = w_q
    enclayer_tf.attention.w_query.bias = b_q
    enclayer_tf.attention.w_key.kernel = w_k
    enclayer_tf.attention.w_key.bias = b_k
    enclayer_tf.attention.w_value.kernel = w_v
    enclayer_tf.attention.w_value.bias = b_v
    enclayer_tf.attention.ff.kernel = w_ff
    enclayer_tf.attention.ff.bias = b_ff

    # MHA2
    w_q = enclayer_pt.conditioned_attention.w_query.weight.detach().numpy()
    b_q = enclayer_pt.conditioned_attention.w_query.bias.detach().numpy().T
    w_k = enclayer_pt.conditioned_attention.w_key.weight.detach().numpy()
    b_k = enclayer_pt.conditioned_attention.w_key.bias.detach().numpy().T
    w_v = enclayer_pt.conditioned_attention.w_value.weight.detach().numpy()
    b_v = enclayer_pt.conditioned_attention.w_value.bias.detach().numpy().T

    w_ff = enclayer_pt.conditioned_attention.ff.weight.detach().numpy().T
    b_ff = enclayer_pt.conditioned_attention.ff.bias.detach().numpy().T

    enclayer_tf.conditioned_attention.w_query.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.conditioned_attention.w_key.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.conditioned_attention.w_value.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.conditioned_attention.ff.build((batch_size, max_sequence_len, d_model))

    enclayer_tf.conditioned_attention.w_query.kernel = w_q
    enclayer_tf.conditioned_attention.w_query.bias = b_q
    enclayer_tf.conditioned_attention.w_key.kernel = w_k
    enclayer_tf.conditioned_attention.w_key.bias = b_k
    enclayer_tf.conditioned_attention.w_value.kernel = w_v
    enclayer_tf.conditioned_attention.w_value.bias = b_v
    enclayer_tf.conditioned_attention.ff.kernel = w_ff
    enclayer_tf.conditioned_attention.ff.bias = b_ff

    # GEGLU
    w1_pt = enclayer_pt.position_wise_feed_forward_layer.w_1.weight.detach().numpy()
    w2_pt = enclayer_pt.position_wise_feed_forward_layer.w_2.weight.detach().numpy()

    b1_pt = enclayer_pt.position_wise_feed_forward_layer.w_1.bias.detach().numpy().T
    b2_pt = enclayer_pt.position_wise_feed_forward_layer.w_2.bias.detach().numpy().T

    enclayer_tf.position_wise_feed_forward_layer.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.position_wise_feed_forward_layer.w_1.build((batch_size, max_sequence_len, d_model))
    dff = 2 * 4 * d_model // 3
    enclayer_tf.position_wise_feed_forward_layer.w_2.build((batch_size, max_sequence_len, dff))

    enclayer_tf.position_wise_feed_forward_layer.w_1.kernel = w1_pt
    enclayer_tf.position_wise_feed_forward_layer.w_1.bias = b1_pt
    enclayer_tf.position_wise_feed_forward_layer.w_2.kernel = w2_pt
    enclayer_tf.position_wise_feed_forward_layer.w_2.bias = b2_pt

    w3_pt = enclayer_pt.position_wise_feed_forward_layer.w_3.weight.detach().numpy()
    b3_pt = enclayer_pt.position_wise_feed_forward_layer.w_3.bias.detach().numpy().T
    enclayer_tf.position_wise_feed_forward_layer.w_3.build((batch_size, max_sequence_len, d_model))
    enclayer_tf.position_wise_feed_forward_layer.w_3.kernel = w3_pt
    enclayer_tf.position_wise_feed_forward_layer.w_3.bias = b3_pt

    output_tf = enclayer_tf(input_tensor_1, input_tensor_2, None, None)[0]

    if pt_channel_axis == -1:
        output_pt = enclayer_pt(torch.from_numpy(input_tensor_1), torch.from_numpy(input_tensor_2), None, None)[0]
    else:
        input_tensor_1 = torch.transpose(torch.from_numpy(input_tensor_1), 1, 2)
        input_tensor_2 = torch.transpose(torch.from_numpy(input_tensor_2), 1, 2)
        output_pt = enclayer_pt(input_tensor_1, input_tensor_2, None, None)[0]
        output_pt = torch.transpose(output_pt, 1, 2)

    # print(output_pt)
    # print(output_tf)

    print('Are the AntheDecoder TF == PT?', np.allclose(output_pt.detach().numpy(), output_tf.numpy(),
                                                        rtol=1.e-4, atol=1.e-7))
