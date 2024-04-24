import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

def get_loss(path, nb_epochs, logging_freq, first_chunk, valid=False):
    for epoch in range(first_chunk, nb_epochs, logging_freq):
        if epoch == first_chunk:
            loss_hist_vRd = np.load(
                path + "data/loss_hist_vRd_{}.npy".format(epoch))
            loss_hist_L2 = np.load(
                path + "data/loss_hist_L2_{}.npy".format(epoch))
            if valid:
                loss_hist_vRd_valid = np.load(
                    path + "data/loss_hist_vRd_valid_{}.npy".format(epoch))
                loss_hist_L2_valid = np.load(
                    path + "data/loss_hist_L2_valid_{}.npy".format(epoch))
        else:
            loss_hist_vRd = np.concatenate((loss_hist_vRd, np.load(
                path + "data/loss_hist_vRd_{}.npy".format(epoch))))
            loss_hist_L2 = np.concatenate((loss_hist_L2, np.load(
                path + "data/loss_hist_L2_{}.npy".format(epoch))))
            if valid:
                loss_hist_vRd_valid = np.concatenate((loss_hist_vRd_valid, np.load(
                    path + "data/loss_hist_vRd_valid_{}.npy".format(epoch))))
                loss_hist_L2_valid = np.concatenate((loss_hist_L2_valid, np.load(
                    path + "data/loss_hist_L2_valid_{}.npy".format(epoch))))

    if valid:
        return loss_hist_vRd, loss_hist_L2, loss_hist_vRd_valid, loss_hist_L2_valid
    else:
        return loss_hist_vRd, loss_hist_L2


def get_meas(path, name, nb_epochs, logging_freq, first_chunk, valid=False):
    for epoch in trange(first_chunk, nb_epochs, logging_freq):
        # print(epoch)
        if epoch == first_chunk:
            hist_h = np.load(path + "data/" + name + "_h_{}.npy".format(epoch))
            hist_o = np.load(path + "data/" + name + "_o_{}.npy".format(epoch))
            if valid:
                hist_h_valid = np.load(
                    path + "data/" + name + "_h_valid_{}.npy".format(epoch))
                hist_o_valid = np.load(
                    path + "data/" + name + "_o_valid_{}.npy".format(epoch))
        else:
            hist_h = np.concatenate((hist_h, np.load(
                path + "data/" + name + "_h_{}.npy".format(epoch))))
            hist_o = np.concatenate((hist_o, np.load(
                path + "data/" + name + "_o_{}.npy".format(epoch))))
            if valid:
                hist_h_valid = np.concatenate((hist_h_valid, np.load(
                    path + "data/" + name + "_h_valid_{}.npy".format(epoch))))
                hist_o_valid = np.concatenate((hist_o_valid, np.load(
                    path + "data/" + name + "_o_valid_{}.npy".format(epoch))))

    if valid:
        return hist_h, hist_o, hist_h_valid, hist_o_valid

    return hist_h, hist_o


def get_spikes_chunk(path, chunk_id):
    print(chunk_id)
    spk_hist_h = np.load(path + "data/spk_hist_h_{}.npy".format(chunk_id))
    spk_hist_o = np.load(path + "data/spk_hist_o_{}.npy".format(chunk_id))

    return spk_hist_h, spk_hist_o


def get_mv_spikes(path, nb_epochs, logging_freq):
    for epoch in range(logging_freq-1, nb_epochs, logging_freq):
        if epoch == logging_freq-1:
            spk_hist_h = np.load(path + "data/spk_hist_h_{}.npy".format(epoch))
            spk_hist_o = np.load(path + "data/spk_hist_o_{}.npy".format(epoch))
        else:
            spk_hist_h = np.concatenate((spk_hist_h, np.load(
                path + "data/spk_hist_h_{}.npy".format(epoch))))
            spk_hist_o = np.concatenate((spk_hist_o, np.load(
                path + "data/spk_hist_o_{}.npy".format(epoch))))

    mean_sh = np.mean(spk_hist_h, axis=1)
    mean_so = np.mean(spk_hist_o, axis=1)
    var_sh = np.var(spk_hist_h, axis=1)
    var_so = np.var(spk_hist_o, axis=1)

    return mean_sh, mean_so, var_sh, var_so


def plot_spikes(x_datas, spk_hs, spk_os, methods, names, data_id=0, epoch_id=-1, trial_id=0, seed_id=0):
    text_props = {'ha': 'center', 'va': 'center', 'fontsize': 8}

    fig, ax = plt.subplots(3, 3, figsize=(
        5, 5), dpi=250, sharex=True, sharey=True)

    for i, method in enumerate(methods):
        ax[0][i].imshow(np.transpose(spk_os[method][seed_id, epoch_id, trial_id, data_id, :, :]),
                        cmap=plt.cm.gray_r, aspect="equal")
        ax[0][i].set_title(names[method], fontsize=8)
        ax[0][i].axis("off")
        ax[1][i].imshow(np.transpose(spk_hs[method][seed_id, epoch_id, trial_id, data_id, :, :]),
                        cmap=plt.cm.gray_r, aspect="equal")
        ax[1][i].axis("off")
        ax[2][i].imshow(np.transpose(x_datas[method][seed_id, data_id]),
                        cmap=plt.cm.gray_r, aspect="equal")
        ax[2][i].axis("off")

    ax[0][0].text(-0.15, 0.5, "Output", text_props, color="black",
                  transform=ax[0][0].transAxes, fontsize=8, rotation=90)
    ax[1][0].text(-0.15, 0.5, "Hidden", text_props, color="black",
                  transform=ax[1][0].transAxes, fontsize=8, rotation=90)
    ax[2][0].text(-0.15, 0.5, "Input", text_props, color="black",
                  transform=ax[2][0].transAxes, fontsize=8, rotation=90)

    sns.despine()
    plt.tight_layout()
    return fig, ax


def plot_spikes_epochs(x_datas, spk_hs, spk_os, methods, names, data_id=0, trial_id=0):
    text_props = {'ha': 'center', 'va': 'center', 'fontsize': 8}

    fig, ax = plt.subplots(3, 2, figsize=(
        4.3, 7.3), dpi=250, sharex=True, sharey=True)

    for i, method in enumerate(methods):
        ax[0][i].imshow(np.transpose(spk_os[method][trial_id, data_id, :, :]),
                        cmap=plt.cm.gray_r, aspect="equal")
        ax[0][i].set_title(names[method])
        # ax[0][i].axis("off")
        ax[1][i].imshow(np.transpose(spk_hs[method][trial_id, data_id, :, :]),
                        cmap=plt.cm.gray_r, aspect="equal")
        # ax[1][i].axis("off")
        ax[2][i].imshow(np.transpose(x_datas[method]),
                        cmap=plt.cm.gray_r, aspect="equal")
        # ax[2][i].axis("off")

        ax[1][i].set_xlabel("Time [ms]")
        ax[0][i].set_yticks([])
        ax[0][i].set_xticks([])
        ax[1][i].set_yticks([])
        ax[1][i].set_xticks([])        
        ax[2][i].set_yticks([])
        ax[2][i].set_xticks([])

    ax[0][0].set_ylabel("Output neurons")
    ax[1][0].set_ylabel("Hidden neurons")
    ax[2][0].set_ylabel("Input neurons")
    
    # ax[0][0].text(-0.15, 0.5, "Output neurons", text_props, color="black",
    #               transform=ax[0][0].transAxes, fontsize=8, rotation=90)
    # ax[1][0].text(-0.15, 0.5, "Hidden neurons", text_props, color="black",
    #               transform=ax[1][0].transAxes, fontsize=8, rotation=90)

    # ax[2][0].text(-0.15, 0.5, "Input neurons", text_props, color="black",
    #               transform=ax[2][0].transAxes, fontsize=8, rotation=90)
    


    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
    return fig, ax
