import jax
import jax.numpy as jnp
from jax import vmap
from jax import jit
from functools import partial


def get_initial_weights(
    key, nb_inputs, nb_hidden, nb_outputs, tau_syn, tau_mem, fi, target_sigma_u=1
):
    eps_hat = tau_syn**2 / (2 * (tau_syn + tau_mem))
    weight_scale_0 = target_sigma_u / jnp.sqrt(nb_inputs * fi * eps_hat)
    weight_scale_1 = target_sigma_u / jnp.sqrt(nb_hidden * fi * eps_hat)

    key, subkey = jax.random.split(key)
    w0 = weight_scale_0 * jax.random.normal(key=subkey, shape=(nb_inputs, nb_hidden))

    key, subkey = jax.random.split(key)
    w1 = weight_scale_1 * jax.random.normal(key=subkey, shape=(nb_hidden, nb_outputs))

    return [w0, w1], key


@partial(jit, static_argnums=(), backend="cpu")
def weighted_spikes(x, w):
    """weight the input spikes with the corresponding synaptic weights (and sum them up)

    Args:
        x (ndarray): input spike trains
        w (ndarray): synaptic weights

    Returns:
        ndarray: the weighted input spikes
    """
    return jnp.matmul(x, w)


get_weighted_spikes = vmap(weighted_spikes, (0, None), 0)


@partial(jit, static_argnums=(), backend="cpu")
def sigmoid(x, delta_u, p0, beta):
    """compute a sigmoid of the input x"""
    return 1 / (1 + jnp.exp(-beta * x))


@partial(jit, static_argnums=(), backend="cpu")
def exp_spike(x, delta_u, p0, beta=0):
    """returns an output spike based on an exponential spike probability given the membrane potential"""
    res = p0 * jnp.exp(x / delta_u)
    return res


@partial(jit, static_argnums=(), backend="cpu")
def _det_spike_fn(x, k, d, p, b, s):
    """returns an output spike using a hard threshold"""
    out = jnp.where(x > 0, 1, 0)
    return out, x + 1


@partial(jit, static_argnums=(), backend="cpu")
def _stoch_spike_fn(x, key, delta_u, p0, beta, sigm=True):
    thr = jax.random.uniform(shape=x.shape, key=key)
    p = jax.lax.cond(sigm, sigmoid, exp_spike, x, delta_u, p0, beta)

    out = jnp.where(p > thr, 1, 0)
    return out, p


@partial(jit, static_argnums=(), backend="cpu")
def spike_fn(x, key, delta_u, p0, beta=10, stoch=True, sigm=True):
    """Defines how spikes are generated

    Args:
        x (array): membrane potential minus threshold
        key (int, optional): random key (used for stochastic spike generation). Defaults to 0.
        beta (int, optional): $\beta$ (used for stochastic spike generation). Defaults to 10.
        stoch (int, optional): Whether using stochastic (True) or deterministic (False) spike generation. Defaults to True.

    Returns:
        _type_: spikes
    """

    out, p = jax.lax.cond(
        stoch, _stoch_spike_fn, _det_spike_fn, x, key, delta_u, p0, beta, sigm
    )
    return out, p


@partial(jit, static_argnums=(), backend="cpu")
def execute_timestep(t, val):
    (
        syn,
        mem,
        spk,
        p,
        syn_nw,
        mem_nw,
        h,
        inp,
        key,
        tau_mem,
        tau_syn,
        delta_u,
        p0,
        beta,
        theta,
        eps_0,
        dt,
        reset_mode,
        stoch,
        sigm,
    ) = val

    mthr = mem[t] - theta
    key, subkey = jax.random.split(key)
    res = spike_fn(mthr, subkey, delta_u, p0, beta, stoch, sigm)
    spk = spk.at[t].set(res[0])
    p = p.at[t].set(res[1])

    dcy_mem = jnp.exp(-dt / tau_mem)
    dcy_syn = jnp.exp(-dt / tau_syn)
    scl_mem = 1 - dcy_mem

    syn = syn.at[t + 1].set(dcy_syn * syn[t] + h[t])
    # mem = mem.at[t+1].set((dcy_mem * mem[t] + scl_mem *
    #                       syn[t]) - (theta * spk[t]))

    mem = jax.lax.cond(
        reset_mode, mult_reset, mult_reset_at_same_t, mem, dcy_mem, spk, scl_mem, syn, t
    )

    # # clip mem at 1
    # mem = jax.lax.cond(clip_mode, clip_mem, no_clip_mem, mem, t)

    syn_nw = syn_nw.at[t + 1].set(dcy_syn * syn_nw[t] + inp[t])
    mem_nw = mem_nw.at[t + 1].set(dcy_mem * mem_nw[t] + scl_mem * syn_nw[t])

    return (
        syn,
        mem,
        spk,
        p,
        syn_nw,
        mem_nw,
        h,
        inp,
        key,
        tau_mem,
        tau_syn,
        delta_u,
        p0,
        beta,
        theta,
        eps_0,
        dt,
        reset_mode,
        stoch,
        sigm,
    )


def mult_reset(mem, dcy_mem, spk, scl_mem, syn, t):
    mem = mem.at[t + 1].set((dcy_mem * mem[t] + scl_mem * syn[t]) * (1 - spk[t]))
    return mem


def mult_reset_at_same_t(mem, dcy_mem, spk, scl_mem, syn, t):
    mem = mem.at[t + 1].set(dcy_mem * mem[t] * (1 - spk[t]) + scl_mem * syn[t])
    return mem


def clip_mem(mem, t):
    mem = mem.at[t + 1].set(jnp.where(mem[t + 1] > 1, 1, mem[t + 1]))
    return mem


def no_clip_mem(mem, t):
    return mem


@partial(
    jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), backend="cpu"
)
def simulate_single(
    h,
    inp,
    key,
    nb_steps,
    nb_in,
    nb_neurons,
    tau_mem,
    tau_syn,
    delta_u,
    p0,
    beta,
    theta,
    eps_0,
    dt,
    reset_mode,
    stoch,
    sigm=True,
):
    """prepare the simulation of one single neuron

    Args:
        h (ndarray): the weighted summed input spikes
        inp (ndarray): the unweighted summed input
        key (_type_): jax random key
        nb_steps (int): number of simulation time steps
        nb_neurons (int): number of neurons
        tau_mem (float): membrane time constant
        tau_syn (float): synaptic time constant
        delta_u (float): $\delta u$
        p0 (float): $p_0$
        beta (float): $\beta$ (steepness of the sigmoid)
        theta (float): threshold
        eps_0 (float): $\epsilon_0$
        dt (float): simulation time step
        reset_mode (bool): whether to use multiplicative reset or multiplicative reset at the same time step
        stoch (bool): whether to use stochastic or deterministic spike generation

    Returns:
        _type_: (syn, mem, spk, p, syn_nw, mem_nw, h, inp, key, tau_mem,
                 tau_syn, delta_u, p0, beta, theta, eps_0, dt, reset_mode, stoch)
        tuple of ndarrays: tuple of membrane potential, spike train, probability of spiking, and unweighted membrane potential for the simulated time steps
    """

    syn = jnp.zeros((nb_steps, nb_neurons))
    mem = jnp.zeros((nb_steps, nb_neurons))
    p = jnp.zeros((nb_steps, nb_neurons))
    spk = jnp.zeros((nb_steps, nb_neurons))

    syn_nw = jnp.zeros((nb_steps, nb_in))
    mem_nw = jnp.zeros((nb_steps, nb_in))

    (
        syn,
        mem,
        spk,
        p,
        syn_nw,
        mem_nw,
        h,
        inp,
        key,
        tau_mem,
        tau_syn,
        delta_u,
        p0,
        beta,
        theta,
        eps_0,
        dt,
        reset_mode,
        stoch,
        sigm,
    ) = jax.lax.fori_loop(
        0,
        nb_steps,
        execute_timestep,
        (
            syn,
            mem,
            spk,
            p,
            syn_nw,
            mem_nw,
            h,
            inp,
            key,
            tau_mem,
            tau_syn,
            delta_u,
            p0,
            beta,
            theta,
            eps_0,
            dt,
            reset_mode,
            stoch,
            sigm,
        ),
    )

    return mem, spk, p, mem_nw


simulate = vmap(
    simulate_single,
    (
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
)


@partial(
    jit,
    static_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    backend="cpu",
)
def run_2l(
    inp,
    w,
    key,
    nb_steps,
    nb_inputs,
    nb_hidden,
    nb_outputs,
    batch_size,
    tau_mem,
    tau_syn,
    delta_uh,
    delta_uo,
    p0,
    beta_h,
    beta_o,
    theta,
    eps_0,
    dt,
    reset_mode,
    stoch,
    sigm=True,
):
    """Run a two layer SNN

    Args:
        inp (array): Input data
        w (list of 2 arrays): hidden layer and output weights
        key (RNGKey): Jax random key
        nb_steps (int): number of time steps
        nb_hidden (int): number of hidden neurons
        nb_outputs (int): number of output neurons
        batch_size (int): number of samples per batch
        tau_mem (float): membrane time constant
        tau_syn (float): synaptic time constant
        delta_uh (float): $\Delta u_h$
        delta_uo (float): $\Delta u_o$
        p0 (float): $\rho_0$
        theta (float): spiking threshold
        eps_0 (float): $\epsilon_0$
        dt (float): time step

    Returns:
        tuple of three list of two arrays: the membrane potential, output spikes and probability of spiking for both layers
    """

    mem_tot = []
    spk_tot = []
    p_tot = []
    mem_tot_nw = []

    # first layer
    h1 = get_weighted_spikes(inp, w[0])
    inp1 = get_weighted_spikes(inp, jnp.eye(nb_inputs))

    key_batch = jax.random.split(key, num=batch_size + 1)
    key = key_batch[-1]
    mem_rec, spk_rec, p_rec, mem_rec_nw = simulate(
        h1,
        inp1,
        key_batch[:-1],
        nb_steps,
        nb_inputs,
        nb_hidden,
        tau_mem,
        tau_syn,
        delta_uh,
        p0,
        beta_h,
        theta,
        eps_0,
        dt,
        reset_mode,
        stoch,
        sigm,
    )

    mem_tot.append(mem_rec)
    spk_tot.append(spk_rec)
    p_tot.append(p_rec)
    mem_tot_nw.append(mem_rec_nw)

    # Second layer
    h2 = get_weighted_spikes(spk_rec, w[1])
    inp2 = get_weighted_spikes(spk_rec, jnp.eye(nb_hidden))

    key_batch = jax.random.split(key, num=batch_size + 1)
    key = key_batch[-1]
    mem_rec, spk_rec, p_rec, mem_rec_nw = simulate(
        h2,
        inp2,
        key_batch[:-1],
        nb_steps,
        nb_hidden,
        nb_outputs,
        tau_mem,
        tau_syn,
        delta_uo,
        p0,
        beta_o,
        theta,
        eps_0,
        dt,
        reset_mode,
        stoch,
        sigm,
    )

    mem_tot.append(mem_rec)
    spk_tot.append(spk_rec)
    p_tot.append(p_rec)
    mem_tot_nw.append(mem_rec_nw)

    return mem_tot, spk_tot, p_tot, mem_tot_nw, key


# @partial(jit, static_argnums=(), backend="cpu")
# def get_loss_out(l1, l2):
#     l = []
#     for o in range(l1.shape[-1]):
#         for h in range(l2.shape[-1]):
#             l.append(jnp.sum(l1[:, o]*l2[:, h]))
#     l = jnp.stack(l)
#     l = jnp.transpose(l.reshape(l1.shape[-1], l2.shape[-1]))
#     return l


# b_get_loss_out = vmap(get_loss_out)


# def run_3l(inp, w, key, nb_steps, nb_hidden, nb_hidden2, nb_outputs, batch_size, tau_mem, tau_syn, delta_uh, delta_uo, p0, theta, eps_0, dt):
#     # first layer
#     h1 = get_weighted_spikes(inp, w[0])

#     key_batch = jax.random.split(key, num=batch_size+1)
#     key = key_batch[-1]
#     mem_rec, spk_rec, p_rec = simulate(
#         h1, key_batch[:-1], nb_steps, nb_hidden, tau_mem, tau_syn, delta_uh, p0, theta, eps_0, dt)

#     # Second layer
#     h2 = get_weighted_spikes(spk_rec, w[1])

#     key_batch = jax.random.split(key, num=batch_size+1)
#     key = key_batch[-1]
#     mem_rec, spk_rec, p_rec = simulate(
#         h2, key_batch[:-1], nb_steps, nb_hidden2, tau_mem, tau_syn, delta_uh, p0, theta, eps_0, dt)

#     # Third layer
#     h3 = get_weighted_spikes(spk_rec, w[2])

#     key_batch = jax.random.split(key, num=batch_size+1)
#     key = key_batch[-1]
#     mem_rec, spk_rec, p_rec = simulate(
#         h3, key_batch[:-1], nb_steps, nb_outputs, tau_mem, tau_syn, delta_uo, p0, theta, eps_0, dt)

#     return spk_rec, key
