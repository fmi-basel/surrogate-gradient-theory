import os

from datetime import datetime
import logging
import wandb

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import stork

from stork.models import RecurrentSpikingModel
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from stork.connections import Connection
from stork.generators import StandardGenerator
from stork.initializers import (
    FluctuationDrivenCenteredNormalInitializer,
    DistInitializer,
)
from stork.datasets import HDF5Dataset, DatasetView
from stork.layers import ConvLayer


# A logger for this file
log = logging.getLogger(__name__)


def prepare_data_randman(cfg: DictConfig):
    """Create and prepare the randman dataset for training. This dataset generates random manifolds in a high
     dimensional space, and samples of mainfolds have to be classified to the correct mainfold.
     Each neuron spikes only `nb_spikes` times , so all information is contained in timing of spikes.

    Args:
        cfg (DictConfig): Contains all the necessary parameters
        - dim_manifold: dimension of the manifold
        - nb_classes: number of classes (manifolds)
        - nb_units: Number of neurons
        - nb_steps: Number of time steps
        - step_frac: fraction of time steps filled with sample, versus silent steps
        - nb_spikes: Numbers of spikes per neurons
        - alpha: roughness of the manifolds
        - seed: random seed of the dataset

    Returns:
        tuple: a tuple of four RasDatasets (train, valid, test, same), split (0.8, 0.1, 0.1), where same contains only
         one sample multiple times
    """
    data, labels = stork.datasets.make_tempo_randman(
        dim_manifold=cfg.dataset.dim_manifold,
        nb_classes=cfg.dataset.nb_classes,
        nb_units=cfg.dataset.nb_inputs,
        nb_steps=cfg.dataset.nb_time_steps,
        step_frac=cfg.dataset.step_frac,
        nb_samples=cfg.dataset.nb_samples,
        nb_spikes=cfg.dataset.nb_spikes,
        alpha=cfg.dataset.alpha_randman,
        seed=cfg.dataset.randmanseed,
    )

    ds_kwargs = dict(
        nb_steps=cfg.dataset.nb_time_steps,
        nb_units=cfg.dataset.nb_inputs,
        time_scale=1.0,
    )

    # Split into train, test and validation set
    datasets = [
        stork.datasets.RasDataset(ds, **ds_kwargs)
        for ds in stork.datasets.split_dataset(
            data, labels, splits=[0.8, 0.1, 0.1], shuffle=False
        )
    ]
    ds_train, ds_valid, ds_test = datasets
    same_dataset = DatasetView(ds_test, [0] * 400)

    return ds_train, ds_valid, ds_test, same_dataset


def prepare_data_shd(cfg: DictConfig):
    """Prepare and load the SHD dataset.

    Args:
        cfg (DictConfig): config containing all the important parameters
        - nb_steps: number of simulation time_steps
        - nb_inputs: Number of input units
        - validation_split: fraction of train set, that should be used for validation

    Returns:
        tuple:  a tuple of four RasDatasets (train, valid, test, same), where same contains only
         one sample multiple times
    """
    gen_kwargs = dict(
        nb_steps=cfg.dataset.nb_steps,
        time_scale=cfg.dataset.time_scale / cfg.dataset.time_step,
        unit_scale=cfg.dataset.unit_scale,
        nb_units=cfg.dataset.nb_inputs,
        preload=True,
        precompute_dense=False,
        unit_permutation=None,
    )
    print(gen_kwargs, cfg.dataset.validation_split)
    train_dataset = HDF5Dataset(
        os.path.join(cfg.dataset.datadir, "shd_train.h5"), **gen_kwargs
    )

    # Split into train and validation set
    mother_dataset = train_dataset
    elements = np.arange(len(mother_dataset))
    np.random.shuffle(elements)

    split = int(cfg.dataset.validation_split * len(mother_dataset))
    valid_dataset = DatasetView(mother_dataset, elements[split:])
    train_dataset = DatasetView(mother_dataset, elements[:split])

    test_dataset = HDF5Dataset(
        os.path.join(cfg.dataset.datadir, "shd_test.h5"), **gen_kwargs
    )

    elements = np.arange(len(test_dataset))
    np.random.shuffle(elements)

    same_dataset = DatasetView(test_dataset, [elements[-1]] * 400)

    return train_dataset, valid_dataset, test_dataset, same_dataset


def prepare_data_ssc(cfg: DictConfig):
    gen_kwargs = dict(
        nb_steps=cfg.dataset.nb_steps,
        time_scale=cfg.dataset.time_scale / cfg.dataset.time_step,
        unit_scale=cfg.dataset.unit_scale,
        nb_units=cfg.dataset.nb_inputs,
        preload=True,
        precompute_dense=False,
        unit_permutation=None,
    )
    print(gen_kwargs)
    train_dataset = HDF5Dataset(
        os.path.join(cfg.dataset.datadir, "ssc_train.h5"), **gen_kwargs
    )
    train_dataset = shuffle_dataset(train_dataset)

    valid_dataset = HDF5Dataset(
        os.path.join(cfg.dataset.datadir, "ssc_valid.h5"), **gen_kwargs
    )
    valid_dataset = shuffle_dataset(valid_dataset)

    test_dataset = HDF5Dataset(
        os.path.join(cfg.dataset.datadir, "ssc_test.h5"), **gen_kwargs
    )
    test_dataset = shuffle_dataset(test_dataset)

    elements = np.arange(len(test_dataset))
    np.random.shuffle(elements)

    same_dataset = DatasetView(test_dataset, [elements[-1]] * 400)

    return train_dataset, valid_dataset, test_dataset, same_dataset


def shuffle_dataset(dataset):
    elements = np.arange(len(dataset))
    np.random.shuffle(elements)
    dataset = DatasetView(dataset, elements)

    return dataset


def anneal(act_fn, cfg: DictConfig):
    old_sg_params = act_fn.surrogate_params
    old_en_params = act_fn.escape_noise_params

    if cfg.method.name != "superspike":
        act_fn.escape_noise_params["beta"] = old_en_params["beta"] + cfg.anneal_step

    act_fn.surrogate_params["beta"] = old_sg_params["beta"] + cfg.anneal_step

    return act_fn


def select_afn(cfg: DictConfig, using_other=False):
    method_name = cfg.method.name
    if using_other:
        method_name = cfg.method.other

    act_fn = stork.activations.CustomSpike

    if method_name == "superspike":
        act_fn.escape_noise_type = "step"
        act_fn.escape_noise_params = {}
        act_fn.surrogate_params = {"beta": cfg.beta}

        if cfg.sigm:
            log.info("using SigmoidSuperSpike")
            act_fn.surrogate_type = "sigmoid"
        else:
            log.info("using SuperSpike")
            act_fn.surrogate_type = "SuperSpike"

    elif method_name == "multilayerspiker":
        act_fn.surrogate_params = {}
        act_fn.surrogate_type = "MultilayerSpiker"

        if cfg.sigm:
            log.info("using SigmoidMultilayerSpikerSpike")
            act_fn.escape_noise_type = "sigmoid"
            act_fn.escape_noise_params = {"beta": cfg.beta}
        else:
            log.info("using MultilayerSpikerSpike")
            act_fn.escape_noise_type = "exponential"
            act_fn.escape_noise_params = {"p0": cfg.p0, "delta_u": cfg.delta_uh}

    elif method_name == "stochasticsuperspike":
        act_fn.escape_noise_type = "sigmoid"
        act_fn.escape_noise_params = {"beta": cfg.beta}
        act_fn.surrogate_params = {"beta": cfg.beta}

        if cfg.sigm:
            log.info("using SigmoidStochasticSigmoidSpike")
            act_fn.surrogate_type = "sigmoid"
        else:
            log.info("using SigmoidStochasticSuperSpike")
            act_fn.surrogate_type = "SuperSpike"

    elif method_name == "exponentialstochasticsuperspike":
        act_fn.escape_noise_type = "exponential"
        act_fn.escape_noise_params = {"p0": cfg.p0, "delta_u": cfg.delta_uh}
        act_fn.surrogate_params = {"beta": cfg.beta}

        if cfg.sigm:
            log.info("using ExponentialStochasticSigmoidSpike")
            act_fn.surrogate_type = "sigmoid"
        else:
            log.info("using ExponentialStochasticSuperSpike")
            act_fn.surrogate_type = "SuperSpike"
    elif method_name == "scaled_sigmoid":
        act_fn.escape_noise_type = "sigmoid"
        act_fn.escape_noise_params = {"beta": cfg.beta}
        act_fn.surrogate_params = {"beta": cfg.beta}

        if cfg.sigm:
            log.info("using Fake Sigmoid stochastic")
            act_fn.surrogate_type = "scaled_sigmoid"

    elif method_name == "realstochasticsuperspike":
        act_fn.escape_noise_type = "SuperSpike"
        act_fn.escape_noise_params = {"beta": cfg.beta}
        act_fn.surrogate_params = {"beta": cfg.beta}

        if cfg.sigm:
            log.info("using SuperStochasticSigmoiSpike")
            act_fn.surrogate_type = "sigmoid"
        else:
            log.info("using SuperStochasticSuperSpike")
            act_fn.surrogate_type = "SuperSpike"

    else:
        raise ValueError(
            "chosen activation function is not supported: " + str(cfg.method.name)
        )
    # print(vars(act_fn))
    return act_fn


def set_up_model(cfg: DictConfig, device, dtype, wandb_handle=None):
    #########################################################
    # Select the activation function
    #########################################################

    act_fn = select_afn(cfg, using_other=False)
    log.info(act_fn)

    #########################################################
    # Select the loss function
    #########################################################

    if cfg.loss_type == "mot_ce":
        loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()
        log.info("using max over time cross entropy loss")
    else:
        raise ValueError("chosen loss function is not supported: " + str(cfg.loss_type))

    #########################################################
    # Select the optimizer
    #########################################################

    # log.info(cfg.optimizer)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD
        optimizer_kwargs = dict(lr=cfg.method.lr, momentum=0)
        log.info("using sgd")
    elif cfg.optimizer == "smorms3":
        opt = stork.optimizers.SMORMS3
        optimizer_kwargs = dict(lr=cfg.method.lr)
        log.info("using smorms3")
    elif cfg.optimizer == "smorms4":
        opt = stork.optimizers.SMORMS4
        optimizer_kwargs = dict(lr=cfg.method.lr)
        log.info("using smorms4")
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam
        optimizer_kwargs = dict(lr=cfg.method.lr)
        log.info("using Adam")

    generator = StandardGenerator(nb_workers=4)

    log.info(opt)
    log.info(optimizer_kwargs)

    #########################################################
    # Regularizer setup
    #########################################################
    upperBoundL2Strength = cfg.regularizer.upperBoundL2Strength
    upperBoundL2Threshold = cfg.regularizer.upperBoundL2Threshold

    # Define regularizer list
    regs = []
    if cfg.regularizer.dims != False:
        dims = list(cfg.regularizer.dims)
    else:
        dims = False

    regUB = stork.regularizers.UpperBoundL2(
        upperBoundL2Strength, threshold=upperBoundL2Threshold, dims=dims
    )
    regs.append(regUB)

    log.info(
        "using regularizer: Thr:"
        + str(upperBoundL2Threshold)
        + ", Strength:"
        + str(upperBoundL2Strength)
        + ", dims:"
        + str(dims)
    )

    # lower bound regularizer
    lowerBoundL2Strength = cfg.regularizer.lowerBoundL2Strength
    lowerBoundL2Threshold = cfg.regularizer.lowerBoundL2Threshold

    regLB = stork.regularizers.LowerBoundL2(
        lowerBoundL2Strength, threshold=lowerBoundL2Threshold, dims=False
    )

    regs.append(regLB)
    print(
        "using regularizer: Thr:",
        lowerBoundL2Threshold,
        ", Strength:",
        lowerBoundL2Strength,
        ", dims:",
        False,
    )

    #########################################################
    # Initializer of the synaptic weights setup
    #########################################################

    initializer = FluctuationDrivenCenteredNormalInitializer(
        sigma_u=cfg.method.target_sigma_u,
        nu=cfg.dataset.nu,
        timestep=cfg.dataset.dt,
        alpha=cfg.dataset.alpha,
    )

    print("Xi:", initializer.xi, 1 / initializer.xi)

    readout_initializer = DistInitializer(
        dist=torch.distributions.Normal(0, 1), scaling="1/sqrt(k)"
    )

    #########################################################
    # Set up the model
    #########################################################

    model = RecurrentSpikingModel(
        cfg.batch_size, cfg.dataset.nb_steps, cfg.dataset.nb_inputs, device, dtype
    )

    # Input Layer
    input_shape = (1, cfg.dataset.nb_inputs)
    input_group = model.add_group(InputGroup(input_shape))

    upstream_group = input_group

    # Hidden Layers
    neuron_group = LIFGroup

    neuron_kwargs = {
        "tau_mem": cfg.tau_mem,
        "tau_syn": cfg.tau_syn,
        "activation": act_fn,
        "diff_reset": cfg.diff_reset,
    }

    recurrent_kwargs = {
        "kernel_size": cfg.dataset.rec_kernel_size,
        "stride": cfg.dataset.rec_stride,
        "padding": cfg.dataset.rec_padding,
    }

    for layer_idx in range(cfg.dataset.nb_hidden_layers):
        # Generate Layer name and config
        layer_name = str("ConvLayer") + " " + str(layer_idx)

        # Make layer
        layer = ConvLayer(
            name=layer_name,
            model=model,
            input_group=upstream_group,
            kernel_size=int(np.array(cfg.dataset.kernel_size)[layer_idx]),
            stride=int(np.array(cfg.dataset.stride)[layer_idx]),
            padding=int(np.array(cfg.dataset.padding)[layer_idx]),
            nb_filters=int(np.array(cfg.dataset.nb_filters)[layer_idx]),
            recurrent=cfg.dataset.rec,
            neuron_class=neuron_group,
            neuron_kwargs=neuron_kwargs,
            recurrent_connection_kwargs=recurrent_kwargs,
            regs=regs,
        )

        # Initialize Parameters
        initializer.initialize(layer)

        # Set output as input to next layer
        upstream_group = layer.output_group

        log.info(layer_name + " added to model")

    # Readout Layer
    readout_group = model.add_group(
        ReadoutGroup(
            cfg.dataset.nb_classes,
            tau_mem=cfg.tau_readout,
            tau_syn=cfg.tau_syn,
            initial_state=-1e-3,
        )
    )

    readout_connection = model.add_connection(
        Connection(upstream_group, readout_group, flatten_input=True)
    )

    # Initialize readout connection
    readout_initializer.initialize(readout_connection)

    #########################################################
    # Add monitors
    #########################################################

    # add monitors
    # for i in range(cfg.dataset.nb_hidden_layers):
    #     model.add_monitor(stork.monitors.SpikeCountMonitor(model.groups[1 + i]))

    for i in range(cfg.dataset.nb_hidden_layers):
        model.add_monitor(stork.monitors.StateMonitor(model.groups[1 + i], "out"))

    model.configure(
        input=input_group,
        output=readout_group,
        loss_stack=loss_stack,
        generator=generator,
        optimizer=opt,
        optimizer_kwargs=optimizer_kwargs,
        time_step=cfg.dataset.dt,
        anneal_start=cfg.anneal_start,
        anneal_step=cfg.anneal_step,
        anneal_interval=cfg.anneal_interval,
        wandb=wandb_handle,
    )

    return model, act_fn


def monitor_model(cfg: DictConfig, model, data, path, same=True, wandb=None, ep=0):
    print("monitor model")
    res = model.monitor(data)

    # evaluate firing rate and Fano factor
    if same and cfg.monitor_fano > 0:
        print("#" * 30, "Firing rates", "#" * 30)

        firing_rates = []
        mean_over_trials = []
        var_over_trials = []
        fano = []

        for i in range(cfg.dataset.nb_hidden_layers):
            ts = torch.sum(res[i], dim=[-3, -2, -1])
            firing_rates.append(
                ts / cfg.dataset.duration / model.groups[i + 1].nb_units
            )

            if cfg.binned:
                print("binned fano factor")
                n_bins = res[i].size(1) // cfg.monitor_fano
                rest = res[i].size(1) % cfg.monitor_fano

                # Calculate the size of the new tensor after binning timesteps
                new_timestep_size = res[i].size(1) // n_bins

                # Reshape the tensor to bin the timesteps
                binned_tensor = res[i][:, :-rest, :].view(
                    res[i].size(0),
                    n_bins,
                    new_timestep_size,
                    res[i].size(2),
                    res[i].size(3),
                )
                # Sum along the binned timestep dimension to aggregate the values
                binned_tensor = binned_tensor.sum(dim=2)

                mot = torch.mean(binned_tensor, dim=0)
                mean_over_trials.append(mot)

                vot = torch.var(binned_tensor, dim=0)
                var_over_trials.append(vot)

                fano.append(np.nanmean(vot / mot))

            else:
                print("moving average fano factor")
                # Instead of binning, use a moving average
                print("res", res[i].shape)
                # compute moving average along second dimension
                moving_average = torch.stack(
                    [
                        torch.sum(res[i][:, j : j + cfg.monitor_fano, :, :], dim=1)
                        for j in range(0, res[i].size(1) - cfg.monitor_fano)
                    ],
                    dim=1,
                )
                # moving_average = torch.stack(nl, dim=1)
                print("mav", moving_average.shape)

                mot = torch.mean(moving_average, dim=0)
                mean_over_trials.append(mot)

                vot = torch.var(moving_average, dim=0)
                var_over_trials.append(vot)

                fano.append(np.nanmean(vot / mot))

        log.info("mean fano factors (spikes): " + str([f for f in fano]))
        log.info(
            "fano factors (firing rates): "
            + str([(torch.var(fr) / torch.mean(fr)).item() for fr in firing_rates])
        )
        log.info("firing rates: " + str([torch.mean(fr).item() for fr in firing_rates]))

        wandb.log(
            {
                "firing_rate-hid_" + str(j): torch.mean(fr).item()
                for j, fr in enumerate(firing_rates)
            },
            step=ep,
        )
        wandb.log({"fano-hid_" + str(j): f for j, f in enumerate(fano)}, step=ep)

    if cfg.plot == "plot":
        if not same:
            log.info("plotting two classes")

            stork.plotting.plot_activity(
                model,
                data=data,
                figsize=(5, 5),
                dpi=150,
                point_alpha=0.3,
                pal=["#008CA5", "#EBB400"],
                pos=(0, 0),
                off=(0.0, -0.05),
            )

            w_dir = os.getcwd()
            file_path = os.path.join(w_dir, path + "_two.png")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close()

            log.info("plotting snapshot")

        plt.figure(dpi=150)
        stork.plotting.plot_activity_snapshot(
            model,
            data=data,
            nb_samples=cfg.plotting.nb_samples,
            figsize=(10, 5),
            dpi=250,
            point_alpha=cfg.plotting.point_alpha,
            pal=sns.color_palette(cc.glasbey, n_colors=cfg.dataset.nb_classes),
        )

        w_dir = os.getcwd()
        file_path = os.path.join(w_dir, path + ".png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        plt.close()

        if same:
            log.info("->")
            log.info("plotting spikes")

            np.random.seed(cfg.plotting.sample_seed)
            fig, axs = plt.subplots(
                1,
                cfg.plotting.nb_samples,
                figsize=(8, 1.5),
                dpi=150,
                sharex=True,
                sharey=True,
            )
            for ax in axs:
                stork.plotting.plot_activity_over_trials(
                    model,
                    data=data,
                    ax=ax,
                    point_alpha=cfg.plotting.point_alpha_trials,
                    point_size=2,
                    layer_idx=cfg.plotting.idx_layer,
                    neuron_idx=np.random.randint(
                        0, model.groups[cfg.plotting.idx_layer + 1].nb_units
                    ),
                    nolabel=True,
                ),
                ax.set_xlabel("time step")

            axs[0].set_ylabel("trial")
            plt.tight_layout()

            w_dir = os.getcwd()
            file_path = os.path.join(w_dir, path + "_spikes.png")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close()


def train(
    cfg: DictConfig,
    model,
    train_dataset,
    valid_dataset,
    results,
    epoch_chunk=10000000,
    offset=0,
):
    nb_epochs = min(cfg.nb_epochs, epoch_chunk)
    history = model.fit(
        train_dataset,
        valid_dataset,
        nb_epochs=nb_epochs,
        verbose=False,
        logger=log,
        log_interval=cfg.logging_freq,
        monitor_spikes=cfg.monitor_spikes,
        anneal=cfg.anneal,
        offset=offset,
    )

    results = {}

    results["train_loss"] = history["loss"].tolist()
    results["train_acc"] = history["acc"].tolist()
    results["valid_loss"] = history["val_loss"].tolist()
    results["valid_acc"] = history["val_acc"].tolist()

    return model, results


def evaluate_training(
    cfg: DictConfig, model, test_data, results, other=False, wandb=None, ep=0
):
    scores = model.evaluate(test_data).tolist()
    if other:
        results["test_loss_other"], _, results["test_acc_other"] = scores
    else:
        results["test_loss"], _, results["test_acc"] = scores
    log.info(results)

    for key, value in results.items():
        if "test" in key:
            wandb.log({key: value}, step=ep)

    if cfg.plot == "plot":
        fig, ax = plt.subplots(2, 2, figsize=(5, 6), dpi=150, sharex=True)

        for i, n in enumerate(["train_loss", "train_acc", "valid_loss", "valid_acc"]):
            if i < 2:
                a = ax[0][i]
            else:
                a = ax[1][i - 2]

            a.plot(results[n], color="black")
            a.set_xlabel("Epochs")
            a.set_ylabel(n)

        ax[0, 1].set_ylim(0, 1)
        ax[1, 1].set_ylim(0, 1)

        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "training.png"))
        plt.close()

        print("Test loss: ", results["test_loss"])
        print("Test acc.: ", results["test_acc"])
    return results


##############################################################################################
# MAIN
##############################################################################################


@hydra.main(version_base=None, config_path="config_stork")
def main(cfg: DictConfig):
    """Main function to train an SNN according to the given config

    Args:
        cfg (DictConfig): Contains all the configs

    Raises:
        ValueError: For invalid configs
    """
    # Update config nb_steps
    nb_steps = int(int(cfg.dataset.duration / cfg.dataset.time_step))
    with open_dict(cfg):
        cfg.dataset.nb_steps = nb_steps

    log.info(OmegaConf.to_yaml(cfg))

    if cfg.using_wandb:
        wrun = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            settings=wandb.Settings(start_method="thread"),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.method.name + "-s=" + str(cfg.seed) + "-dr=" + str(cfg.diff_reset)  + "-lr=" + str(cfg.method.lr) ,
            # mode="disabled"
        )
        print(wandb.config)

    print("#" * 30 + "\n # DATASET\n" + "#" * 30)

    # Set up seeds
    log.info("set up seeds: %i", cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch.float

    if cfg.dataset.name == "shd":
        log.info("prepare data shd")
        train_data, valid_data, test_data, same_data = prepare_data_shd(cfg)
    elif cfg.dataset.name == "randman":
        log.info("prepare data randman")
        train_data, valid_data, test_data, same_data = prepare_data_randman(cfg)
    elif cfg.dataset.name == "ssc":
        log.info("prepare data ssc")
        train_data, valid_data, test_data, same_data = prepare_data_ssc(cfg)
    else:
        raise ValueError("chosen dataset is not supported: " + str(cfg.dataset.name))

    print("#" * 30 + "\n# MODEL\n" + "#" * 30)

    log.info("set up model")
    if cfg.using_wandb:
        print("using wandb")
        wandb_handle = wrun
    else:
        print("not using wand")
        wandb_handle = None

    model, act_fn = set_up_model(cfg, device, dtype, wandb_handle=wandb_handle)

    results = {}

    if cfg.monitor_chunks:
        epoch_chunk = cfg.epoch_chunk
    else:
        epoch_chunk = cfg.nb_epochs

    for i in range(0, cfg.nb_epochs, epoch_chunk):
        print("#" * 30 + f"\n# EVALUATE - Epoch {i}\n" + "#" * 30)

        monitor_model(
            cfg,
            model,
            test_data,
            f"Plots/{i}",
            same=False,
            wandb=wandb_handle,
            ep=i + 1,
        )
        if cfg.method.name != "superspike":
            monitor_model(
                cfg,
                model,
                same_data,
                f"Plots/{i}_same",
                same=True,
                wandb=wandb_handle,
                ep=i + 1,
            )

        print("#" * 15 + "\n# TRAIN\n" + "#" * 15)
        log.info("training")
        log.info(datetime.now())
        start = datetime.now()

        model, results = train(
            cfg,
            model,
            train_data,
            valid_data,
            results,
            epoch_chunk=epoch_chunk,
            offset=i,
        )

        end = datetime.now()
        duration = end - start
        print(duration)
        log.info("train duration: %s", duration)
        # log.info("Printing results")
        # log.info(results)

    print("#" * 30 + "\n# EVALUATE\n" + "#" * 30)

    monitor_model(
        cfg,
        model,
        test_data,
        "Plots/after",
        same=False,
        wandb=wandb_handle,
        ep=cfg.nb_epochs + 2,
    )
    evaluate_training(
        cfg, model, test_data, results, wandb=wandb_handle, ep=cfg.nb_epochs + 2
    )
    log.info("Evaluated results")

    if cfg.method.name != "superspike":
        monitor_model(
            cfg,
            model,
            same_data,
            "Plots/after_same",
            same=True,
            wandb=wandb_handle,
            ep=cfg.nb_epochs + 2,
        )
        log.info("Evaluated same")

    # select other activation function
    act_fn_new = select_afn(cfg, using_other=True)
    for i in range(1, len(model.groups) - 1):
        model.groups[i].spk_nl = act_fn_new.apply
        print(model.groups[i].spk_nl)

    monitor_model(
        cfg,
        model,
        test_data,
        "Plots/after_otherafn",
        same=False,
        wandb=wandb_handle,
        ep=cfg.nb_epochs + 2,
    )
    results = evaluate_training(
        cfg,
        model,
        test_data,
        results,
        other=True,
        wandb=wandb_handle,
        ep=cfg.nb_epochs + 2,
    )
    log.info("Evaluated results (other_activation)")

    if cfg.method.other != "superspike":
        monitor_model(
            cfg,
            model,
            same_data,
            "Plots/same_after_otherafn",
            same=True,
            wandb=wandb_handle,
            ep=cfg.nb_epochs + 2,
        )
        log.info("Evaluated same (other_activation)")

    wrun.finish()
    print("*" * 100, "DONE", "*" * 100)


if __name__ == "__main__":
    main()
