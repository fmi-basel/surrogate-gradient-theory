import matplotlib.pyplot as plt
import seaborn as sns
from utils import weight_update
import jax.numpy as jnp


def make_plot_hid(x_data, z_data, mem, spk, p, filtered_input, filtered_hidden, dw, batch_id, nb_steps, i_neuron_id, h_neuron_id, o_neuron_id, wbound):
    """plot the quantities used for the hidden layer weight updates in the same manner as in the Gardener, Sporea & Grünig paper from 2015

    Args:
        x_data (ndarray): the input spikes
        z_data (ndarray): the output spikes
        mem (list of ndarray): the membrane potential of the hidden and the output layer
        spk (list of ndarray): the spike trains of the hidden and the output layer
        p (list of ndarray): the probability of spiking of the hidden and the output layer
        filtered_input (ndarray): the input spikes filtered with the kernel epsilon
        filtered_hidden (ndaray): the hidden layer spikes filtered with the kernel epsilon
        dw (ndarray): the momentary weight update
        batch_id (int): the batch id
        nb_steps (int): number of simulation time steps
        i_neuron_id (int): the input neuron id
        h_neuron_id (int): the hidden neuron id
        o_neuron_id (int): the output neuron id
        wbound (float): limit for the weight update plot y axis

    Returns:
        fig, ax: the matplotlib figure
    """

    fig, ax = plt.subplots(2, 3, figsize=(9, 3), dpi=250, sharex=True)

    ############################################################################################################################
    # Input related
    ############################################################################################################################

    ax[0][0].plot(5*x_data[batch_id, :, i_neuron_id] -
                  1, color="blue", lw=1, label="$X_i$")
    ax[0][0].set_ylim(0, 1)

    ax[1][0].plot(filtered_input[batch_id, :, i_neuron_id],
                  color="blue", lw=1, label="$(X_i*\epsilon)$", zorder=-2)

    ############################################################################################################################
    # Hidden layer related
    ############################################################################################################################

    ax[0][1].plot(mem[0][batch_id, :, h_neuron_id] + spk[0][batch_id,
                  :, h_neuron_id]*5, color="red", lw=1, label="$u_h$", zorder=-2, alpha=0.5)
    ax[0][1].plot(mem[0][batch_id, :, h_neuron_id],
                  color="red", lw=1, label="$u_h$", zorder=-2)
    ax[0][1].plot(p[0][batch_id, :, h_neuron_id], color="black",
                  lw=1, label="$p_h$", zorder=-2, alpha=0.7)
    ax[0][1].plot([0, nb_steps], [1, 1], '--', color="silver", zorder=-5)
    ax[0][1].set_ylim(-1, 2)

    ax[1][1].plot(filtered_hidden[batch_id, i_neuron_id, :, h_neuron_id],
                  color="red", lw=1, label="$(Y_h(X_i*\epsilon)*\epsilon)$", zorder=-2)

    ############################################################################################################################
    # Output related
    ############################################################################################################################

    ax[0][2].plot(mem[1][batch_id, :, o_neuron_id]+5*spk[1][batch_id,
                  :, o_neuron_id], color="green", lw=1, label="$u_o$", zorder=-2, alpha=0.5)
    ax[0][2].plot(mem[1][batch_id, :, o_neuron_id],
                  color="green", lw=1, label="$u_o$", zorder=-2)
    ax[0][2].plot([0, nb_steps], [1, 1], '--', color="silver", zorder=-3)
    ax[0][2].plot(5*z_data[batch_id, :, o_neuron_id]-2, "--",
                  color="black", lw=1, label="$Z_o$", alpha=0.5)
    ax[0][2].set_ylim(-1, 2)

    ax[1][2].plot(dw[batch_id, :, i_neuron_id, h_neuron_id],
                  color="green", lw=1, label="$\Delta w_{hi}$", zorder=-2)
    ax[1][2].plot(5*z_data[batch_id, :, o_neuron_id]-2, "--",
                  color="black", lw=1, label="$Z_o$", alpha=0.5)
    ax[1][2].set_ylim(-wbound, wbound)

    for a in ax:
        for b in a:
            b.set_xlim(0, nb_steps)
            b.legend(loc="upper right")
    sns.despine()
    plt.tight_layout()

    return fig, ax


def make_plot_out(z_data, filtered_hidden, dw, mem, spk, p, batch_id, nb_steps, i_neuron_id, h_neuron_id, o_neuron_id, wbound):
    """plot the quantities used for the output layer weight updates in the same manner as in the Gardener, Sporea & Grünig paper from 2015

    Args:
        z_data (ndarray): target spiketrain
        filtered_hidden (ndarray): hidden layer spikes filtered with kernel epsilon
        dw (ndarray): output layer weight update
        mem (list of ndarray): the membrane potential of the hidden and the output layer
        spk (list of ndarray): the spike trains of the hidden and the output layer
        p (list of ndarray): the probability of spiking of the hidden and the output layer
        batch_id (int): the batch id
        nb_steps (int): number of simulation time steps
        i_neuron_id (int): the input neuron id
        h_neuron_id (int): the hidden neuron id
        o_neuron_id (int): the output neuron id
        wbound (float): limit for the weight update plot y axis

    Returns:
        fig, ax: the matplotlib figure
    """

    fig, ax = plt.subplots(2, 2, figsize=(6, 3), dpi=250, sharex=True)

    ############################################################################################################################
    # Hidden layer related
    ############################################################################################################################

    ax[0][0].plot(mem[0][batch_id, :, h_neuron_id] + spk[0][batch_id,
                  :, h_neuron_id]*5, color="red", lw=1, label="$u_h$", zorder=-2, alpha=0.5)
    ax[0][0].plot(mem[0][batch_id, :, h_neuron_id],
                  color="red", lw=1, label="$u_h$", zorder=-2)
    ax[0][0].plot(p[0][batch_id, :, h_neuron_id], color="black",
                  lw=1, label="$p_h$", zorder=-2, alpha=0.7)
    ax[0][0].plot([0, nb_steps], [1, 1], '--', color="silver", zorder=-5)
    ax[0][0].set_ylim(-1, 2)

    ax[1][0].plot(filtered_hidden[batch_id, :, h_neuron_id],
                  color="red", lw=1, label="$(Y_h*\epsilon)$", zorder=-2)

    ############################################################################################################################
    # Output related
    ############################################################################################################################

    ax[0][1].plot(mem[1][batch_id, :, o_neuron_id]+5*spk[1][batch_id,
                  :, o_neuron_id], color="green", lw=1, label="$u_o$", zorder=-2, alpha=0.5)
    ax[0][1].plot(mem[1][batch_id, :, o_neuron_id],
                  color="green", lw=1, label="$u_o$", zorder=-2)
    ax[0][1].plot([0, nb_steps], [1, 1], '--', color="silver", zorder=-3)
    ax[0][1].plot(5*z_data[batch_id, :, o_neuron_id]-2, "--",
                  color="black", lw=1, label="$Z_o$", alpha=0.5)
    ax[0][1].set_ylim(-1, 2)

    ax[1][1].plot(dw[batch_id, :, i_neuron_id, h_neuron_id],
                  color="green", lw=1, label="$\Delta w_{oh}$", zorder=-2)
    ax[1][1].plot(5*z_data[batch_id, :, o_neuron_id]-2, "--",
                  color="black", lw=1, label="$Z_o$", alpha=0.5)
    ax[1][1].set_ylim(-wbound, wbound)

    for a in ax:
        for b in a:
            b.set_xlim(0, nb_steps)
            b.legend(loc="upper right")
    sns.despine()
    plt.tight_layout()

    return fig, ax


def plot_spk_over_epochs(spk, target, batch_id, nb_neurons, nb_epochs, n_skip):
    """plot the spikes over all epochs for the number of neurons

    Args:
        spk (ndarray): the history of output spikes
        target (ndarray): the target spike train
        batch_id (int): the batch id
        nb_neurons (int): number of neurons to plot
        nb_epochs (int): number of epochs
        n_skip (int): which number of epochs to plot (e.g. 10 means plotting every 10th epoch)

    Returns:
        fig, ax: matplotlib figure handles
    """
    fig, axs = plt.subplots(1, nb_neurons, figsize=(
        nb_neurons, 2), dpi=250, sharey=True, sharex=True)
    if nb_neurons == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.imshow(spk[::n_skip, batch_id, :, i],
                  aspect="auto", cmap="binary", label="out")
        if target is not None:
            ax.scatter(jnp.where(target[batch_id, :, i])[0], jnp.ones(len(jnp.where(
                target[batch_id, :, i])[0]))*nb_epochs//n_skip*1.01, color="red", s=3, label="target", zorder=10)

        ax.invert_yaxis()

    plt.yticks(jnp.linspace(0, nb_epochs//n_skip, 5),
               jnp.linspace(0, nb_epochs, 5).astype(int))
    axs[0].set_ylabel("epoch")
    axs[(nb_neurons-1)//2].set_xlabel("time [a.u.]")
    sns.despine()

    return fig, axs
