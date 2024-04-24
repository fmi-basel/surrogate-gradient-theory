import numpy as np
import os
import jax.numpy as jnp


def space_image(z_data):
    zeros = jnp.zeros_like(z_data)

    test = [[]]

    for i, (d, z) in enumerate(zip(z_data[0], zeros[0])):
        test[0].append(z)
        test[0].append(d)

    return jnp.array(test)


def dense2ras(densespikes, time_step=1e-3, concatenate_trials=True):
    """ Returns ras spike format list of tuples (time, neuronid) or dense input data.

    Args:
    densespikes -- Either a matrix (time, neuron) of spikes or a rank 3 tensor (trial, time, neuron)
    time_step -- Time in seconds assumed per temporal biin

    Returns: 
    A list of spikes in ras forma or when multiple trials are given as list of lists of spikes unless
    concatenate_trials is set to true in which case all trials will be concatenated.
    """

    if len(densespikes.shape) == 3:
        trials = []
        for spk in densespikes:
            trials.append(dense2ras(spk))
        if concatenate_trials:
            ras = []
            td = densespikes.shape[1]  # trial duration
            for k, trial in enumerate(trials):
                toff = np.zeros(trial.shape)
                toff[:, 0] = k * td * time_step
                ras.append(trial + toff)
            return np.concatenate(ras, axis=0)
        else:
            return trials
    elif len(densespikes.shape) == 2:
        ras = []
        aw = np.argwhere(densespikes > 0.0)
        for t, i in aw:
            ras.append((t * time_step, int(i)))
        return np.array(ras)
    else:
        print("Input array shape not understood.")
        raise ValueError


def create_folder(dir, name):
    path = os.path.join(dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_jnp(path, data):
    with open(path, 'ab') as f:
        jnp.save(f, data)


def create_and_jnpsave(dir, name, data):
    path = create_folder(dir, name)
    save_jnp(path, data)
