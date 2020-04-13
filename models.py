import math
from typing import List

import jax.numpy as np
from jax import lax, random
from jax.nn.initializers import glorot_normal, normal, uniform
from jax.nn.initializers import xavier_uniform
import jax.nn as nn


def Dropout(rate):
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.

    Arguments:
        rate (float): Probability of keeping and element.
    """
    def init_fun(rng, input_shape):
        return input_shape, ()
    def apply_fun(params, inputs, is_training, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        keep = random.bernoulli(rng, rate, inputs.shape)
        outs = np.where(keep, inputs / rate, 0)
        # if not training, just return inputs and discard any computation done
        out = lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
        return out
    return init_fun, apply_fun


def GraphAttentionLayer(out_dim, dropout, residual=False):
    """
    Layer constructor function for a Graph Attention layer.
    """
    _, drop_fun = Dropout(dropout)
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4 = random.split(rng, 4)
        stdv = 1. / math.sqrt(out_dim)
        W_init = uniform(stdv)
        # projection
        W = W_init(k1, (input_shape[-1], out_dim))

        a_init = xavier_uniform()
        a1 = a_init(k2, (out_dim, 1))
        a2 = a_init(k3, (out_dim, 1))

        return output_shape, (W, a1, a2)
       
    def apply_fun(params, x, adj, rng, activation=nn.elu, is_training=False, 
                  **kwargs):
        W, a1, a2 = params
        x = np.dot(x, W)

        f_1 = np.dot(x, a1) 
        f_2 = np.dot(x, a2)
        logits = f_1 + f_2.T
        coefs = nn.softmax(
            nn.leaky_relu(logits, negative_slope=0.2) + np.where(adj, 0., -1e9))

        k1, k2 = random.split(rng, 2) 
        coefs = drop_fun(None, coefs, is_training=is_training, rng=k1)
        x = drop_fun(None, x, is_training=is_training, rng=k2)

        ret = np.matmul(coefs, x)

        return activation(ret)

    return init_fun, apply_fun


def GAT(nheads: List[int], nhid: List[int], nclass: int, dropout: float,
        residual: bool=False):
    """
    Graph Attention Network model definition.
    """

    init_funs = []
    attn_funs = []
    _, drop_fun = Dropout(dropout)

    nhid += [nclass]
    for layer_i in range(len(nhid)):
        layer_funs, layer_inits = [], []
        for head_i in range(nheads[layer_i]):
            att_init, att_fun = GraphAttentionLayer(nhid[layer_i], 
                                    dropout=dropout,
                                    residual=residual)
            layer_inits.append(att_init)
            layer_funs.append(att_fun)
        attn_funs.append(layer_funs)
        init_funs.append(layer_inits)

    def init_fun(rng, input_shape):
        params = []
        for i, layer_inits in enumerate(init_funs):
            for init_fun in layer_inits:
                rng, layer_rng = random.split(rng)
                layer_shape, param = init_fun(layer_rng, input_shape)
                params.append(param)
            input_shape = layer_shape
            if i < len(init_funs) - 1: # not the last layer
                # multiply by the number of heads
                input_shape = input_shape[:-1] + (input_shape[-1]*len(layer_inits),)
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)

        x = drop_fun(None, x, is_training=is_training, rng=rng)

        for layer_i in range(len(nhid)-1):
            layer_outs = []
            for head_i in range(nheads[layer_i]):
                # TODO: this index could be wrong if each layer has a differnt
                # number of heads
                idx = layer_i * nheads[layer_i] + head_i
                layer_params = params[idx]
                layer_outs.append(attn_funs[layer_i][head_i](
                        layer_params, x, adj, rng=rng, is_training=is_training))
            x = np.concatenate(layer_outs, axis=1)
            x = drop_fun(None, x, is_training=is_training, rng=rng)
        
        # out layer
        layer_outs = []
        for head_i in range(nheads[-1]):
            layer_i = len(nhid)-1
            idx = layer_i * nheads[-2] + head_i
            layer_params = params[idx]
            layer_outs.append(attn_funs[layer_i][head_i](
                    layer_params, x, adj, activation=lambda x: x, rng=rng,
                    is_training=is_training))

        # average last layer heads
        x = np.mean(np.stack(layer_outs), axis=0) 
        return nn.log_softmax(x)

    return init_fun, apply_fun
