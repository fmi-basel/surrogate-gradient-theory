import jax
import hydra
from omegaconf import OmegaConf
import logging
import os

import sys  # noqa

sys.path.append("../")  # noqa
from utils import snn, datasets, stochastic_network, utils  # noqa


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config_experiments")
def main(cfg):

    # device_idx = 0
    # jax.config.update("jax_default_device", jax.devices()[device_idx])

    log.info("====================================================")
    # device_idx = 0
    # jax.config.update("jax_default_device", jax.devices()[device_idx])
    # log.info("Using device: %s", jax.devices()[device_idx])

    print(OmegaConf.to_yaml(cfg))

    # print current working directory
    print(os.getcwd())

    key = jax.random.PRNGKey(cfg.seed)

    log.info("prepare data")
    x_data, z_data, key = datasets.prepare_data(
        data_path=cfg.dataset.data_path,
        key=key,
        nb_steps=cfg.dataset.nb_steps,
        nb_inputs=cfg.dataset.nb_inputs,
        nb_outputs=cfg.dataset.nb_outputs,
        out_prob=cfg.out_prob,
        fi=cfg.fi,
        batch_size=cfg.dataset.batch_size,
        dt=cfg.dt,
        reset_mode=cfg.reset_mode,
    )

    w_dir = os.getcwd()

    utils.create_and_jnpsave(w_dir, "data/x_data.npy", x_data)
    utils.create_and_jnpsave(w_dir, "data/z_data.npy", z_data)

    log.info("generate initial weights")
    w, key = snn.get_initial_weights(
        key,
        cfg.dataset.nb_inputs,
        cfg.dataset.nb_hidden,
        cfg.dataset.nb_outputs,
        cfg.tau_syn,
        cfg.tau_mem,
        cfg.fi,
        cfg.target_sigma_u,
    )

    log.info("create network")
    sto_net = stochastic_network.StochasticNetwork(
        tau_mem=cfg.tau_mem,
        tau_syn=cfg.tau_syn,
        lr_h=cfg.dataset_method.lr_h,
        lr_o=cfg.dataset_method.lr_o,
        nb_steps=cfg.dataset.nb_steps,
        nb_inputs=cfg.dataset.nb_inputs,
        nb_hidden=cfg.dataset.nb_hidden,
        nb_outputs=cfg.dataset.nb_outputs,
        method=cfg.method.sg_mode,
        batch_size=cfg.dataset.batch_size,
        nb_epochs=cfg.nb_epochs,
        nb_trials=cfg.method.nb_trials,
        eps0=cfg.eps0,
        delta_uh=cfg.delta_uh,
        delta_uo=cfg.delta_uo,
        p0=cfg.p0,
        beta_h=cfg.beta_h,
        beta_o=cfg.beta_o,
        theta=cfg.theta,
        save_dw=cfg.save_dw,
        save_w=cfg.save_w,
        save_mem=cfg.save_mem,
        save_spk=cfg.save_spk,
        reset_mode=cfg.reset_mode,
        stoch=cfg.method.stoch,
        sigm=cfg.method.sigm,
        dt=cfg.dt,
        logging_freq=cfg.logging_freq,
        validate_with_other_method=cfg.validate_with_other_method,
    )
    log.info("start training")
    sto_net.train(w, x_data, z_data, key, log)


if __name__ == "__main__":
    main()
