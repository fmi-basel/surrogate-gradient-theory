import jax
import jax.numpy as jnp
from jax import vmap
from jax import jit
from functools import partial

from tqdm import tqdm


@partial(jit, static_argnums=(), backend="cpu")
def eps_kernel(t, eps_0, tau_mem, tau_syn):
    res = eps_0 * (jnp.exp(-t / tau_mem) - jnp.exp(-t / tau_syn))
    if jnp.isscalar(res):
        if t < 0:
            return 0
        else:
            return res
    res = jnp.where(t < 0, 0, res)
    return res


@partial(jit, static_argnums=(2), backend="cpu")
def conv1d_s(x, eps, nb_steps):
    res = jax.scipy.signal.convolve(x, eps)[:nb_steps]
    return res


b_conv_s = vmap(conv1d_s, in_axes=(1, None, None), out_axes=(1))
get_filtered_spiketrain = vmap(b_conv_s, (0, None, None))

get_weighted_filtered_spiketrain = vmap(get_filtered_spiketrain, (0, None, None))


def loop_over_hidden(h, val):
    res, i, filt_inp, hidden = val
    res = res.at[i, :, h].set(filt_inp[:, i] * hidden[:, h])
    return res, i, filt_inp, hidden


@partial(jit, static_argnums=(2, 3, 4), backend="cpu")
def inp_weighted_hidden(hidden, filt_inp, nb_inputs, nb_hidden, nb_steps):
    res = jnp.zeros(shape=(nb_inputs, nb_steps, nb_hidden))
    for i in range(nb_inputs):
        res, _, filt_inp, hidden = jax.lax.fori_loop(
            0, nb_hidden, loop_over_hidden, (res, i, filt_inp, hidden)
        )
    return res


get_inp_weighted_hidden = vmap(inp_weighted_hidden, (0, 0, None, None, None))


def loop_whi_over_inputs(i, val):
    w, w_oh_h, delta_o, filt_w_hid, h = val
    dw = jnp.sum(jnp.sum(w_oh_h * delta_o, axis=1) * filt_w_hid[i, :, h])
    w = w.at[i, h].set(w[i, h] + dw)
    return w, w_oh_h, delta_o, filt_w_hid, h


@partial(jit, static_argnums=(3, 4), backend="cpu")
def get_w_hi_updated_inner(delta_o, filt_w_hid, w_oh, nb_inputs, nb_hidden):
    w = jnp.zeros(shape=(nb_inputs, nb_hidden))
    # over hidden neurons
    for h in range(nb_hidden):
        # over input neurons
        w, w_oh_h, delta_o, filt_w_hid, _ = jax.lax.fori_loop(
            0, nb_inputs, loop_whi_over_inputs, (w, w_oh[h], delta_o, filt_w_hid, h)
        )
    return w


get_w_hi_updated = vmap(get_w_hi_updated_inner, (0, 0, None, None, None))


def loop_who_over_inputs(o, val):
    w, delta_o, filt_hid_h, h = val

    dw = jnp.sum(delta_o[:, o] * filt_hid_h, axis=0)
    w = w.at[h, o].set(w[h, o] + dw)

    return w, delta_o, filt_hid_h, h


@partial(jit, static_argnums=(2, 3), backend="cpu")
def get_w_oh_updated_inner(delta_o, filt_hid, nb_hidden, nb_outputs):
    w = jnp.zeros(shape=(nb_hidden, nb_outputs))
    # over hidden neurons
    for h in range(nb_hidden):
        # over output neurons
        w, delta_o, filt_hid_h, _ = jax.lax.fori_loop(
            0, nb_outputs, loop_who_over_inputs, (w, delta_o, filt_hid[:, h], h)
        )
    return w


get_w_oh_updated = vmap(get_w_oh_updated_inner, (0, 0, None, None))


def get_delta_w0(
    filt_w_hid, delta_o, w, nb_steps, lr_h, delta_uh, nb_inputs, nb_hidden
):
    delta_w0 = []

    for t in tqdm(range(nb_steps)):
        d = jnp.zeros_like(delta_o).at[:, :t, :].set(delta_o[:, :t, :])
        f = jnp.zeros_like(filt_w_hid).at[:, :, :t, :].set(filt_w_hid[:, :, :t, :])
        delta_w0.append(
            lr_h / delta_uh * get_w_hi_updated(d, f, w[1], nb_inputs, nb_hidden)
        )
    delta_w0 = jnp.stack(delta_w0, axis=1)

    return delta_w0


def get_delta_w1(delta_o, filtered_hidden, nb_steps, lr_o, nb_hidden, nb_outputs):
    delta_w1 = []

    for t in tqdm(range(nb_steps)):
        d = jnp.zeros_like(delta_o).at[:, :t, :].set(delta_o[:, :t, :])
        f = jnp.zeros_like(filtered_hidden).at[:, :t, :].set(filtered_hidden[:, :t, :])
        delta_w1.append(lr_o * get_w_oh_updated(d, f, nb_hidden, nb_outputs))
    delta_w1 = jnp.stack(delta_w1, axis=1)
    return delta_w1


def get_update(
    mem_tot_nw,
    z_data,
    epsilon,
    spk_tot,
    sg,
    w,
    nb_steps,
    nb_inputs,
    nb_hidden,
    nb_outputs,
    delta_uo,
):
    filtered_input = mem_tot_nw[0]
    filtered_hidden = mem_tot_nw[1]

    w_hid = get_inp_weighted_hidden(sg, filtered_input, nb_inputs, nb_hidden, nb_steps)

    filt_w_hid = get_weighted_filtered_spiketrain(w_hid, epsilon, nb_steps)

    delta_o = (z_data - spk_tot[1]) / delta_uo  # Their assumption

    dw1 = jnp.average(
        get_w_oh_updated(delta_o, filtered_hidden, nb_hidden, nb_outputs), axis=0
    )
    dw0 = jnp.average(
        get_w_hi_updated(delta_o, filt_w_hid, w[1], nb_inputs, nb_hidden), axis=0
    )

    return dw0, dw1


def get_update_out(mem_tot_nw, z_data, spk_tot, nb_hidden, nb_outputs, delta_uo):
    print("not multiplied by learning rate")
    filtered_input = mem_tot_nw[0]

    delta_o = (z_data - spk_tot[0]) / delta_uo  # Their assumption

    dw1 = jnp.average(
        get_w_oh_updated(delta_o, filtered_input, nb_hidden, nb_outputs), axis=0
    )

    return dw1


@partial(jit, static_argnums=(1, 2), backend="cpu")
def sg_SuperSpike(x, thr, beta=5):
    return 1 / (beta * jnp.absolute(x - thr) + 1) ** 2


sg_SuperSpike_neurons = vmap(sg_SuperSpike, in_axes=(0, None, None))
sg_SuperSpike_times = vmap(sg_SuperSpike_neurons, in_axes=(0, None, None))
get_sg_SuperSpike = vmap(sg_SuperSpike_times, in_axes=(0, None, None))


@partial(jit, static_argnums=(1, 2), backend="cpu")
def sg_SigmoidSpike(x, thr, beta):
    return beta * jnp.exp(-beta * (x - thr)) / ((1 + jnp.exp(-beta * (x - thr))) ** 2)


sg_SigmoidSpike_neurons = vmap(sg_SigmoidSpike, in_axes=(0, None, None))
sg_SigmoidSpike_times = vmap(sg_SigmoidSpike_neurons, in_axes=(0, None, None))
get_sg_SigmoidSpike = vmap(sg_SigmoidSpike_times, in_axes=(0, None, None))


@partial(jit, static_argnums=(1, 2, 3), backend="cpu")
def sg_ExponentialSpike(x, thr, p0=0.01, delta_u=0.133):
    return p0 / delta_u * jnp.exp((x - thr) / delta_u)


sg_ExponentialSpike_neurons = vmap(sg_ExponentialSpike, in_axes=(0, None, None, None))
sg_ExponentialSpike_times = vmap(
    sg_ExponentialSpike_neurons, in_axes=(0, None, None, None)
)
get_sg_ExponentialSpike = vmap(sg_ExponentialSpike_times, in_axes=(0, None, None, None))
