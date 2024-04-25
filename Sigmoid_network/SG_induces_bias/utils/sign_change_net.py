import jax.numpy as jnp
from jax import grad, jit, vmap
from scipy import interpolate

from functools import partial

from tqdm import tqdm

# General Case

#######################################################################################
# ACTIVATION FUNCTIONS
#######################################################################################


@jit
def sigmoid(x, beta=1):
    return 1 / (1 + jnp.exp(-beta * x))


@jit
def grad_sigmoid(x, beta=1):
    return (beta * jnp.exp(-beta * x)) / ((jnp.exp(-beta * x) + 1) ** 2)


#######################################################################################
# evaluate network and gradient while changing parameters along any direction
#######################################################################################


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


def net_get_y_value(w_, x, a=0, b=0, d1=jnp.array([0]), d2=jnp.array([0]), beta=10e6):
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


def net_get_activity(w_, x, beta=10e6):
    """Get one single value of network output for a specific set of parameter

    Args:
        w (array): network weigths (shape=(5,))
        x (float): input
        beta (int): steepness of the sigmoid activation

    Returns:
        array: network outputs along shifting d1 and d2
    """

    w = w_[0]
    v1 = w_[1]
    v2 = w_[2]
    u1 = w_[3]
    u2 = w_[4]

    g = sigmoid(w * x, beta)
    h1 = sigmoid(v1 * g, beta)
    h2 = sigmoid(v2 * g, beta)
    y = sigmoid(u1 * h1 + u2 * h2, beta)

    return g, h1, h2, y


net_get_y_value_over_d1 = vmap(
    net_get_y_value, in_axes=(None, None, 0, None, None, None, None), out_axes=0
)
net_get_y_value_over_d2 = vmap(
    net_get_y_value_over_d1, in_axes=(None, None, None, 0, None, None, None), out_axes=0
)


#######################################################################################
# GRADIENTS
#######################################################################################


def grad_net(w, x, a, b, d1, d2, beta, beta_sg):
    return net_get_grad_value_over_d2(w, x, a, b, d1, d2, beta, beta_sg)


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

#######################################################################################
# get gradient along a given direction


def get_grad_along_direction_single(g, d):
    return jnp.dot(g, d)


get_grad_along_direction_w1 = vmap(
    get_grad_along_direction_single, in_axes=(0, None), out_axes=0
)
get_grad_along_direction = vmap(
    get_grad_along_direction_w1, in_axes=(0, None), out_axes=0
)


########################################################################################################################


def get_int_grads(grads, d1, d2, ref_idx=0):
    grads_d1 = jnp.dot(grads, d1)
    grads_d2 = jnp.dot(grads, d2)
    ref_d1 = jnp.cumsum(grads_d1[ref_idx])
    ref_d2 = jnp.cumsum(grads_d2[:, ref_idx])

    # sum grads along direcion d2
    g1 = jnp.cumsum(grads_d2, axis=0) + ref_d1
    g2 = jnp.cumsum(grads_d1, axis=1) + ref_d2[:, None]

    return g1, g2


#######################################################################################
# evaluate network while changing parameters along unit vectors
#######################################################################################

# Direction U


def net_u(w, v1, v2, u1, u2, x, beta):
    g = sigmoid(w * x, beta)
    h1 = sigmoid(v1 * g, beta)
    h2 = sigmoid(v2 * g, beta)

    y = net_u_get_y_value_over_u2(u1, h1, u2, h2, beta)
    return y


def net_u_get_y_value(u1, h1, u2, h2, beta):
    return sigmoid(u1 * h1 + u2 * h2, beta)


net_u_get_y_value_over_u1 = vmap(
    net_u_get_y_value, in_axes=(0, None, None, None, None), out_axes=0
)
net_u_get_y_value_over_u2 = vmap(
    net_u_get_y_value_over_u1, in_axes=(None, None, 0, None, None), out_axes=0
)


# Direction V


def net_v(w, v1, v2, u1, u2, x, beta):
    g = sigmoid(w * x, beta)
    y = net_v_get_y_value_over_v2(g, v1, v2, u1, u2, beta)
    return y


def net_v_get_y_value(g, v1, v2, u1, u2, beta):
    h1 = sigmoid(v1 * g, beta)
    h2 = sigmoid(v2 * g, beta)
    return sigmoid(u1 * h1 + u2 * h2, beta)


net_v_get_y_value_over_v1 = vmap(
    net_v_get_y_value, in_axes=(None, 0, None, None, None, None), out_axes=0
)
net_v_get_y_value_over_v2 = vmap(
    net_v_get_y_value_over_v1, in_axes=(None, None, 0, None, None, None), out_axes=0
)


# Direction W


def net_w_get_y_value(w, v1, v2, u1, u2, x, beta):
    g = sigmoid(w * x, beta)
    h1 = sigmoid(v1 * g, beta)
    h2 = sigmoid(v2 * g, beta)
    return sigmoid(u1 * h1 + u2 * h2, beta)


net_w = vmap(
    net_w_get_y_value, in_axes=(0, None, None, None, None, None, None), out_axes=0
)


#######################################################################################
# GRADIENTS OVER CHANGING PARAMETERS
#######################################################################################


# Direction U


def get_grad_for_u1(w, v1, v2, u1, u2, x, beta, beta_sg):
    grads = grad_value(jnp.array(w), v1, v2, u1, u2, x, beta, beta_sg)
    return grads[3]


get_grads_over_u1 = vmap(
    get_grad_for_u1, (None, None, None, 0, None, None, None, None), 0
)


def get_gradients_over_u1(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_u1(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


def get_grad_for_u2(w, v1, v2, u1, u2, x, beta, beta_sg):
    grads = grad_value(jnp.array(w), v1, v2, u1, u2, x, beta, beta_sg)
    return grads[4]


get_grads_over_u2 = vmap(
    get_grad_for_u2, (None, None, None, None, 0, None, None, None), 0
)


def get_gradients_over_u2(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_u2(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


def get_grads_over_u2_fixed_u1(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_u2(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


get_grads_over_u2_for_all_u1 = vmap(
    get_grads_over_u2_fixed_u1, (None, None, None, 0, None, None, None, None, 0), 0
)


def get_grads_over_u1_fixed_u2(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_u1(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


get_grads_over_u1_for_all_u2 = vmap(
    get_grads_over_u1_fixed_u2, (None, None, None, None, 0, None, None, None, 0), 0
)


# Direction V


def get_grad_for_v1(w, v1, v2, u1, u2, x, beta, beta_sg):
    grads = grad_value(jnp.array(w), v1, v2, u1, u2, x, beta, beta_sg)
    return grads[1]


get_grads_over_v1 = vmap(
    get_grad_for_v1, (None, 0, None, None, None, None, None, None), 0
)


def get_gradients_over_v1(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_v1(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


def get_grad_for_v2(w, v1, v2, u1, u2, x, beta, beta_sg):
    grads = grad_value(jnp.array(w), v1, v2, u1, u2, x, beta, beta_sg)
    return grads[2]


get_grads_over_v2 = vmap(
    get_grad_for_v2, (None, None, 0, None, None, None, None, None), 0
)


def get_gradients_over_v2(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_v2(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


def get_grads_over_v2_fixed_v1(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_v2(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


get_grads_over_v2_for_all_v1 = vmap(
    get_grads_over_v2_fixed_v1, (None, 0, None, None, None, None, None, None, 0), 0
)


def get_grads_over_v1_fixed_v2(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_v1(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


get_grads_over_v1_for_all_v2 = vmap(
    get_grads_over_v1_fixed_v2, (None, None, 0, None, None, None, None, None, 0), 0
)


# Direction W


def get_grad_for_w(w, v1, v2, u1, u2, x, beta, beta_sg):
    grads = grad_value(jnp.array(w), v1, v2, u1, u2, x, beta, beta_sg)
    return grads[0]


get_grads_over_w = vmap(
    get_grad_for_w, (0, None, None, None, None, None, None, None), 0
)


def get_gradients_over_w(w, v1, v2, u1, u2, x, beta, beta_sg, y):
    gradient = get_grads_over_w(w, v1, v2, u1, u2, x, beta, beta_sg)
    int_gradient = jnp.cumsum(gradient) + y
    return int_gradient, gradient


############################################################################################################
# TRAJECTORY FOR 3D PLOTS
############################################################################################################


# @partial(jit, static_argnums=(3, 4, 5, 6))
def get_trajectory(d1, d2, Z, scalex=4, scaley=4, shiftx=0, shifty=0):
    X, Y = jnp.meshgrid(d1, d2)

    t = jnp.linspace(-5, 5, 100001)
    xt = scalex * jnp.sin(2 * jnp.pi * t / 10) + shiftx
    yt = scaley * jnp.cos(2 * jnp.pi * t / 10) + shifty

    XY = jnp.stack([X.ravel(), Y.ravel()]).T
    S = interpolate.LinearNDInterpolator(XY, Z.ravel())

    xyt = jnp.stack([xt, yt]).T
    St = S(xyt)

    Sd = jnp.cumsum(jnp.sqrt(jnp.sum(jnp.diff(xyt, axis=0) ** 2, axis=1)))

    return xt, yt, t, X, Y, St, Sd
