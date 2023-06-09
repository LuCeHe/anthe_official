import numpy as np
from math import sqrt
import tensorflow as tf


def highestPowerOf2(n):
    return (n & (~(n - 1)))


def factors(n):  # (cf. https://stackoverflow.com/a/15703327/849891)
    j = 2
    while n > 1:
        for i in range(j, int(sqrt(n + 0.05)) + 1):
            if n % i == 0:
                n //= i;
                j = i
                yield i
                break
        else:
            if n > 1:
                yield n;
                break


def get_two_factors(n):
    term1 = 1
    factor_list = list(factors(n))
    ln = len(factor_list)
    if ln == 2:
        return factor_list[0], factor_list[1]
    elif ln == 3:
        return factor_list[0] * factor_list[1], factor_list[2]
    elif ln > 3:
        flist = sorted(factor_list)
        terms = np.ones(2, dtype=int)
        counter = 0
        while len(flist) > 0:
            terms[counter % 2 * (-1) ** (counter // 2) - (counter // 2) % 2] *= flist.pop(0)
            counter += 1
        return terms[0], terms[1]


def get_four_factors(n):
    factor_list = list(factors(n))
    if len(factor_list) == 2:
        return factor_list[0], 1, 1, factor_list[1]
    elif len(factor_list) == 3:
        return factor_list[0], 1, factor_list[1], factor_list[2]
    if len(factor_list) == 4:
        return factor_list[0], factor_list[1], factor_list[2], factor_list[3]
    elif len(factor_list) > 4:
        flist = sorted(factor_list)
        terms = np.ones(4, dtype=int)
        counter = 0
        while len(flist) > 0:
            terms[counter % 4 * (-1) ** (counter // 4) - (counter // 4) % 2] *= flist.pop(0)
            counter += 1
        return terms[0], terms[1], terms[2], terms[3]


def get_three_factors_v2(n):
    factor_list = list(factors(n))
    if len(factor_list) == 2:
        return factor_list[0], 1, factor_list[1]
    elif len(factor_list) == 3:
        return factor_list[0], factor_list[1], factor_list[2]
    elif len(factor_list) > 3:
        flist = sorted(factor_list)
        terms = np.ones(3, dtype=int)
        counter = 0
        while len(flist) > 0:
            terms[counter % 3 * (-1) ** (counter // 3) - (counter // 3) % 2] *= flist.pop(0)
            counter += 1
        return terms[0], terms[1], terms[2]


def get_three_factors(n):
    m = highestPowerOf2(n)
    r = n // m
    ml = int(np.log2(m))
    if r > 1 and m > 1:
        if m > 2:
            result = [2 ** (ml // 2), r, m // (2 ** (ml // 2))]
        else:
            rlist = list(factors(r))
            rlen = len(rlist)
            if rlen == 1:
                result = [2, r, 1]
            else:
                result = [int(np.prod(rlist[:rlen - 1])), 2, rlist[rlen - 1]]
    elif r == 1:
        m = n // 2
        return [2 ** ((ml - 1) // 2), 2, m // (2 ** ((ml - 1) // 2))]
    elif m == 1 and r == n:
        mlist = list(factors(n))
        mlen = len(mlist)
        if mlen == 1:
            result = [1, n, 1]
        elif mlen == 2:
            result = [mlist[0], 1, mlist[1]]
        else:
            result = [mlist[0], int(np.prod(mlist[1:mlen - 1])), mlist[mlen - 1]]
    assert (int(np.prod(result)) == n)
    return result


def get_tc_kernel(self, input_size, output_size, length, bond, ratio, name,
                   initializer, regularizer, constraint, return_tensors_and_einsum_string=False):
    # random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    # name = name + '_' + random_string

    if bond is None or bond < 1:
        if ratio is None:
            r = 0.2
        else:
            r = ratio
        assert r < 1 and r > 0
    kernels = []

    if length == 2:
        input_dims = get_two_factors(input_size)
        output_dims = get_two_factors(output_size)

        bond = max(2, int(input_size * output_size * r
                          / (input_dims[0] * output_dims[0] + input_dims[1] * output_dims[1])))

        kernel_dims = [[input_dims[0], bond, output_dims[0]],
                       [input_dims[1], bond, output_dims[1]]]

        einsum_string = 'ijk,ljr->ilkr'

        true_ratio = input_size * output_size / bond / (
                input_dims[0] * output_dims[0]
                + input_dims[1] * output_dims[1])

    elif length == 3:
        input_dims = get_three_factors(input_size)
        output_dims = get_three_factors(output_size)

        bond = max(2, int((-input_dims[0] * output_dims[0]
                           - input_dims[2] * output_dims[2]
                           + sqrt((input_dims[0] * output_dims[0]
                                   + input_dims[2] * output_dims[2]) ** 2
                                  + 4 * input_dims[1] * output_dims[1]
                                  * input_size * output_size * r))
                          / (2 * input_dims[1] * output_dims[1])))

        kernel_dims = [[input_dims[0], bond, output_dims[0]],
                       [input_dims[1], bond, bond, output_dims[1]], [input_dims[2], bond, output_dims[2]]]

        einsum_string = 'ijk,ljmr,smp->ilskrp'

        true_ratio = input_size * output_size / bond / (
                input_dims[0] * output_dims[0]
                + input_dims[2] * output_dims[2]
                + input_dims[1] * output_dims[1] * bond
        )

    elif length == 4:

        input_dims = get_four_factors(input_size)
        output_dims = get_four_factors(output_size)

        bond = max(2, int(
            -input_dims[0] * output_dims[0] - input_dims[3] * output_dims[3]
            + sqrt((input_dims[0] * output_dims[0] + input_dims[3] * output_dims[3]) ** 2 +
                   4 * (input_dims[1] * output_dims[1] + input_dims[2] * output_dims[2]) *
                   input_size * output_size * r) / (
                               2 * (input_dims[1] * output_dims[1] + input_dims[2] * output_dims[2]))))

        kernel_dims = [[input_dims[0], bond, output_dims[0]],
                       [input_dims[1], bond, bond, output_dims[1]], [input_dims[2], bond, bond, output_dims[2]],
                       [input_dims[3], bond, output_dims[3]]]

        einsum_string = 'ijk,ljmr,smop,tof->ilstkrpf'

        true_ratio = input_size * output_size / bond / (
                input_dims[0] * output_dims[0]
                + input_dims[2] * output_dims[2] * bond
                + input_dims[1] * output_dims[1] * bond
                + input_dims[3] * output_dims[3]
        )


    elif length == -3:
        input_dims = get_three_factors_v2(input_size)
        output_dims = get_three_factors_v2(output_size)

        bond = max(2, int((-input_dims[0] * output_dims[0]
                           - input_dims[2] * output_dims[2]
                           + sqrt((input_dims[0] * output_dims[0]
                                   + input_dims[2] * output_dims[2]) ** 2
                                  + 4 * input_dims[1] * output_dims[1]
                                  * input_size * output_size * r))
                          / (2 * input_dims[1] * output_dims[1])))

        kernel_dims = [[input_dims[0], bond, output_dims[0]],
                       [input_dims[1], bond, bond, output_dims[1]], [input_dims[2], bond, output_dims[2]]]

        einsum_string = 'ijk,ljmr,smp->ilskrp'


    else:
        raise ValueError('TC length greater than 4 not implemented')

    for i in range(len(kernel_dims)):
        kernels.append(self.add_weight(name + '_' + str(i),
                                       shape=kernel_dims[i],
                                       initializer=initializer, regularizer=regularizer,
                                       constraint=constraint, trainable=True, dtype=self.dtype))

    if not return_tensors_and_einsum_string:
        kernel = tf.einsum(einsum_string, *kernels)
        kernel = tf.reshape(kernel, [input_size, output_size])

        return kernel
    else:
        return kernels, einsum_string
