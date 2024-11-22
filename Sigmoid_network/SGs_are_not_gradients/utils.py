# activation function and its gradient
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


@jit
def sigmoid(x, beta=1):
    return 1 / (1 + jnp.exp(-beta * x))


@jit
def grad_sigmoid(x, beta=1):
    return (beta * jnp.exp(-beta * x)) / ((jnp.exp(-beta * x) + 1) ** 2)


# Get network outputs


@partial(jit, static_argnums=(6))
def net_get_y_value(w_, x, a, b, d1, d2, beta):
    """Get one single value of network output for a specific set of parameter

    Args:
        w (array): network weigths (shape=(5,))
        x (float): input
        a (float): current step along direction d1
        b (float): current step along direction d2
        d1 (array): shape of the weigths, parameter direction for change
        d2 (array): shape of the weights, parameter direction for change
        beta (int): steepness of the sigmoid activation

    Returns:
        array: network outputs along shifting d1 and d2
    """

    w = w_[0] + d1[0] * a + d2[0] * b
    v1 = w_[1] + d1[1] * a + d2[1] * b
    v2 = w_[2] + d1[2] * a + d2[2] * b
    u1 = w_[3] + d1[3] * a + d2[3] * b
    u2 = w_[4] + d1[4] * a + d2[4] * b

    g = sigmoid(w * x, beta)
    h1 = sigmoid(v1 * g, beta)
    h2 = sigmoid(v2 * g, beta)
    y = sigmoid(u1 * h1 + u2 * h2, beta)

    return y


net_get_y_value_over_d1 = vmap(
    net_get_y_value, in_axes=(None, None, 0, None, None, None, None), out_axes=0
)
net_get_y_value_over_d2 = vmap(
    net_get_y_value_over_d1, in_axes=(None, None, None, 0, None, None, None), out_axes=0
)


def net(w, x, a, b, d1, d2, beta):
    """Get output values of the networks while shifting parameters along directions d1 and d2 with steps a and b, using beta

    Args:
        w (array): network weigths (shape=(5,))
        x (float): input
        a (array): steps along direction d1
        b (array): steps along direction d2
        d1 (array): shape of the weigths, parameter direction for change
        d2 (array): shape of the weights, parameter direction for change
        beta (int): steepness of the sigmoid activation

    Returns:
        array: network outputs along shifting d1 and d2
    """
    return net_get_y_value_over_d2(w, x, a, b, d1, d2, beta)


# compute network gradient


@jit
def grad_value(w, v1, v2, u1, u2, x, beta, beta_sg):
    """Compute the gradient for a specific set of weights"""
    g = sigmoid(w * x, beta)
    g_sg = grad_sigmoid(w * x, beta_sg)

    h1 = sigmoid(v1 * g, beta)
    h1_sg = grad_sigmoid(v1 * g, beta_sg)
    h2 = sigmoid(v2 * g, beta)
    h2_sg = grad_sigmoid(v2 * g, beta_sg)

    y = sigmoid(u1 * h1 + u2 * h2, beta)
    y_sg = grad_sigmoid(u1 * h1 + u2 * h2, beta_sg)

    dydu1 = h1 * y_sg
    dydu2 = h2 * y_sg

    dydh1 = u1 * y_sg
    dydh2 = u2 * y_sg

    dh1dv1 = g * h1_sg
    dh2dv2 = g * h2_sg

    dh1dg = v1 * h1_sg
    dh2dg = v2 * h2_sg

    dgdw = x * g_sg

    return jnp.array(
        [
            (dydh1 * dh1dg + dydh2 * dh2dg) * dgdw,
            dydh1 * dh1dv1,
            dydh2 * dh2dv2,
            dydu1,
            dydu2,
        ]
    )


def net_get_grad_value(w_, x, a, b, d1, d2, beta, beta_sg):
    w = w_[0] + d1[0] * a + d2[0] * b
    v1 = w_[1] + d1[1] * a + d2[1] * b
    v2 = w_[2] + d1[2] * a + d2[2] * b
    u1 = w_[3] + d1[3] * a + d2[3] * b
    u2 = w_[4] + d1[4] * a + d2[4] * b

    return grad_value(w, v1, v2, u1, u2, x, beta, beta_sg)


net_get_grad_value_over_d1 = vmap(
    net_get_grad_value,
    in_axes=(None, None, 0, None, None, None, None, None),
    out_axes=0,
)
net_get_grad_value_over_d2 = vmap(
    net_get_grad_value_over_d1,
    in_axes=(None, None, None, 0, None, None, None, None),
    out_axes=0,
)


def grad_net(w, x, a, b, d1, d2, beta, beta_sg):
    return net_get_grad_value_over_d2(w, x, a, b, d1, d2, beta, beta_sg)


# get integrated gradients


def get_int_grads(grads, d1, d2, ref_d1=0, ref_d2=0):
    grads_d1 = jnp.dot(grads, d1)
    grads_d2 = jnp.dot(grads, d2)

    # sum grads along direcion
    g1 = jnp.cumsum(grads_d1, axis=1) + ref_d1
    g2 = jnp.cumsum(grads_d2, axis=0) + ref_d2

    return g1, g2


def get_ig(ig1_dict, ig2_dict, beta_sg, lines):
    ig_l1 = jnp.array(ig1_dict[beta_sg][lines[0][:, 0], lines[0][:, 1]])

    ig_l2 = jnp.array(ig2_dict[beta_sg][lines[1][:, 0], lines[1][:, 1]])
    ig_l2 += ig_l1[-1].item() - ig_l2[0].item()

    ig_l3 = jnp.array(ig1_dict[beta_sg][lines[2][:, 0], lines[2][:, 1]])
    ig_l3 += ig_l2[-1].item() - ig_l3[0].item()

    ig_l4 = jnp.array(ig2_dict[beta_sg][lines[3][:, 0], lines[3][:, 1]])
    ig_l4 += ig_l3[-1].item() - ig_l4[0].item()

    ig = jnp.concatenate([ig_l1, ig_l2, ig_l3, ig_l4])

    return ig
