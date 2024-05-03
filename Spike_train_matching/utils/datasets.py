
import jax.numpy as jnp

import jax
import jax.numpy as jnp

import matplotlib.image as mpimg

from . import utils


###########################################################################################################
# For Image Spike-Train matching task
###########################################################################################################


def prepare_data(data_path, key, nb_steps, nb_inputs, nb_outputs, out_prob, fi, batch_size, dt=1e-3, reset_mode=True):

    # Input: generate frozen Poisson spikes
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(key=subkey, shape=(
        batch_size, nb_steps, nb_inputs))
    x_data = jnp.zeros(
        (batch_size, nb_steps, nb_inputs))

    prob = dt * fi
    x_data = x_data.at[mask < prob].set(1)

    # Target output: Image
    print("data_path: ", data_path)
    img = mpimg.imread(data_path)
    mask = jnp.array([[*jnp.transpose(img[:, :, 0])]])
    z_data = jnp.zeros(
        (batch_size, nb_steps, nb_outputs))
    z_data = z_data.at[mask < out_prob].set(1)

    if reset_mode:
        z_data = utils.space_image(z_data)


    return x_data, z_data, key

