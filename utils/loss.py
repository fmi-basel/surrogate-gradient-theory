import jax.numpy as jnp
from jax import vmap

from utils import weight_update


# @partial(jit, static_argnums=(3))
def _van_Rossum_distance(spk1, spk2, epsilon, nb_steps):
    """computes the van Rossum distance between two spiketrains for a given kernel $\epsilon$

    Args:
        spk1 (binary 1darray): first spiketrain
        spk2 (binary 1darray): second spiketrain
        epsilon (1darray): the convolution kernel
        nb_steps (int): the nubmer of timesteps

    Returns:
        float: the distance between the two spiketrains
    """
    f_spk1 = weight_update.conv1d_s(spk1, epsilon, nb_steps)
    f_spk2 = weight_update.conv1d_s(spk2, epsilon, nb_steps)

    dist = (f_spk1 - f_spk2)**2

    return 1/2 * jnp.sum(dist)


_b_van_Rossum_distance = vmap(
    _van_Rossum_distance, in_axes=(1, 1, None, None))
get_van_Rossum_distance = vmap(_b_van_Rossum_distance, (0, 0, None, None))


def _L2_loss(spk1, spk2):
    """computes the L2 distance between two spiketrains

    Args:
        spk1 (binary 1darray): first spiketrain
        spk2 (binary 1darray): second spiketrain

    Returns:
        float: the distance between the two spiketrains
    """

    dist = (spk1 - spk2)**2

    return 1/2 * jnp.sum(dist)


_b_L2_loss = vmap(
    _L2_loss, in_axes=(1, 1))
get_L2_loss = vmap(_b_L2_loss, (0, 0))
