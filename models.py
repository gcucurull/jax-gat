from typing import List

import jax.numpy as np
from jax import lax, random
from jax.nn.initializers import glorot_normal, glorot_uniform
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
        W_init = glorot_uniform()
        # projection
        W = W_init(k1, (input_shape[-1], out_dim))

        a_init = glorot_uniform()
        a1 = a_init(k2, (out_dim, 1))
        a2 = a_init(k3, (out_dim, 1))

        return output_shape, (W, a1, a2)
       
    def apply_fun(params, x, adj, rng, activation=nn.elu, is_training=False, 
                  **kwargs):
        W, a1, a2 = params
        k1, k2, k3 = random.split(rng, 3) 
        x = drop_fun(None, x, is_training=is_training, rng=k1)
        x = np.dot(x, W)

        f_1 = np.dot(x, a1) 
        f_2 = np.dot(x, a2)
        logits = f_1 + f_2.T
        coefs = nn.softmax(
            nn.leaky_relu(logits, negative_slope=0.2) + np.where(adj, 0., -1e9))

        coefs = drop_fun(None, coefs, is_training=is_training, rng=k2)
        x = drop_fun(None, x, is_training=is_training, rng=k3)

        ret = np.matmul(coefs, x)

        return activation(ret)

    return init_fun, apply_fun


def MultiHeadLayer(nheads: int, nhid: int, dropout: float, residual: bool=False,
                   last_layer: bool=False):
    layer_funs, layer_inits = [], []
    for head_i in range(nheads):
        att_init, att_fun = GraphAttentionLayer(nhid, dropout=dropout,
                                residual=residual)
        layer_inits.append(att_init)
        layer_funs.append(att_fun)
    
    def init_fun(rng, input_shape):
        params = []
        for att_init_fun in layer_inits:
            rng, layer_rng = random.split(rng)
            layer_shape, param = att_init_fun(layer_rng, input_shape)
            params.append(param)
        input_shape = layer_shape
        if not last_layer:
            # multiply by the number of heads
            input_shape = input_shape[:-1] + (input_shape[-1]*len(layer_inits),)
        return input_shape, params
    
    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        layer_outs = []
        assert len(params) == nheads
        for head_i in range(nheads):
            layer_params = params[head_i]
            rng, _ = random.split(rng)
            layer_outs.append(layer_funs[head_i](
                    layer_params, x, adj, rng=rng, is_training=is_training))
        if not last_layer:
            x = np.concatenate(layer_outs, axis=1)
        else:
            # average last layer heads
            x = np.mean(np.stack(layer_outs), axis=0)

        return x

    return init_fun, apply_fun


def GAT(nheads: List[int], nhid: List[int], nclass: int, dropout: float,
        residual: bool=False):
    """
    Graph Attention Network model definition.
    """

    init_funs = []
    attn_funs = []

    nhid += [nclass]
    for layer_i in range(len(nhid)):
        last = layer_i == len(nhid) - 1
        layer_init, layer_fun = MultiHeadLayer(nheads[layer_i], nhid[layer_i],
                                    dropout=dropout, residual=residual,
                                    last_layer=last)
        attn_funs.append(layer_fun)
        init_funs.append(layer_init)

    def init_fun(rng, input_shape):
        params = []
        for i, init_fun in enumerate(init_funs):
            rng, layer_rng = random.split(rng)
            layer_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
            input_shape = layer_shape
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, len(attn_funs))

        for i, layer_fun in enumerate(attn_funs):
            x = layer_fun(params[i], x, adj, rng=rngs[i], is_training=is_training)
        
        return nn.log_softmax(x)

    return init_fun, apply_fun
