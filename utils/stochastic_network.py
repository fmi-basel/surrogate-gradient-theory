import jax
import jax.numpy as jnp
import os

from utils import snn, weight_update, loss, utils

from tqdm import tqdm


class StochasticNetwork:
    def __init__(
        self,
        tau_mem,
        tau_syn,
        lr_h,
        lr_o,
        nb_steps,
        nb_inputs,
        nb_hidden,
        nb_outputs,
        method="superspike",
        batch_size=1,
        nb_epochs=1000,
        nb_trials=1,
        eps0=1,
        delta_uh=133e-3,
        delta_uo=13e-3,
        p0=0.6666667,
        beta_h=10,
        beta_o=100,
        theta=1,
        save_dw=True,
        save_w=True,
        save_spk=True,
        save_mem=True,
        reset_mode=False,
        stoch=True,
        sigm=True,
        dt=1e-3,
        logging_freq=100,
        validate_with_other_method=True,
    ):
        self.logging_freq = logging_freq
        self.validate_with_other_method = validate_with_other_method

        self.method = method
        self.stoch = stoch
        self.valid_stoch = not stoch
        print("stoch: ", stoch)
        print("valid_stoch: ", self.valid_stoch)
        self.sigm = sigm
        self.reset_mode = reset_mode

        self.save_mem = save_mem
        self.save_spk = save_spk
        self.save_w = save_w
        self.save_dw = save_dw

        self.lr_h = lr_h
        self.lr_o = lr_o

        self.nb_steps = nb_steps
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.batch_size = batch_size

        self.dt = dt
        self.nb_epochs = nb_epochs
        self.nb_trials = nb_trials

        self.tau_mem = tau_mem
        self.tau_syn = tau_syn

        self.eps0 = eps0
        self.delta_uh = delta_uh
        self.beta_h = beta_h
        self.delta_uo = delta_uo
        self.beta_o = beta_o
        self.p0 = p0
        self.theta = theta
        self.eps_0 = eps0

    def train(self, w, x_data, z_data, key, log):
        l_loss_hist_vRd = []
        l_loss_hist_L2 = []

        if self.validate_with_other_method:
            l_loss_hist_vRd_valid = []
            l_loss_hist_L2_valid = []

        if self.save_dw:
            l_dw_hist_h = []
            l_dw_hist_o = []

        if self.save_w:
            l_w_hist_h = []
            l_w_hist_o = []

        if self.save_spk:
            l_spk_hist_h = []
            l_spk_hist_o = []
            if self.validate_with_other_method:
                l_spk_hist_h_valid = []
                l_spk_hist_o_valid = []

        if self.save_mem:
            l_mem_hist_h = []
            l_mem_hist_o = []

        dw0_mean_hist = []
        dw1_mean_hist = []

        duration = self.nb_steps * self.dt
        t = jnp.linspace(0, duration, self.nb_steps)
        epsilon = weight_update.eps_kernel(t, self.eps_0, self.tau_mem, self.tau_syn)

        for e in tqdm(range(self.nb_epochs)):
            dw0_list = []
            dw1_list = []

            if self.save_spk:
                spk0_hist = []
                spk1_hist = []
                if self.validate_with_other_method:
                    spk0_hist_valid = []
                    spk1_hist_valid = []

            if self.save_mem:
                mem0_hist = []
                mem1_hist = []

            for trial in range(self.nb_trials):
                key, subkey = jax.random.split(key)
                mem_tot, spk_tot, p_tot, mem_tot_nw, key = snn.run_2l(
                    x_data,
                    w,
                    subkey,
                    self.nb_steps,
                    self.nb_inputs,
                    self.nb_hidden,
                    self.nb_outputs,
                    self.batch_size,
                    self.tau_mem,
                    self.tau_syn,
                    self.delta_uh,
                    self.delta_uo,
                    self.p0,
                    self.beta_h,
                    self.beta_o,
                    self.theta,
                    self.eps_0,
                    self.dt,
                    self.reset_mode,
                    self.stoch,
                    self.sigm,
                )

                if self.method == "superspike":
                    sg = weight_update.get_sg_SuperSpike(
                        mem_tot[0], self.theta, self.beta_h
                    )
                elif self.method == "sigmoidspike":
                    sg = weight_update.get_sg_SigmoidSpike(
                        mem_tot[0], self.theta, self.beta_h
                    )
                elif self.method == "multilayerspiker":
                    sg = spk_tot[0]

                dw0, dw1 = weight_update.get_update(
                    mem_tot_nw,
                    z_data,
                    epsilon,
                    spk_tot,
                    sg,
                    w,
                    self.nb_steps,
                    self.nb_inputs,
                    self.nb_hidden,
                    self.nb_outputs,
                    self.delta_uo,
                )

                dw0 = self.lr_h * dw0
                dw1 = self.lr_o * dw1

                dw0_list.append(dw0)
                dw1_list.append(dw1)

                if self.save_spk:
                    # print("spk_tot[0]:", spk_tot[0].shape)
                    spk0_hist.append(spk_tot[0])
                    spk1_hist.append(spk_tot[1])

                if self.save_mem:
                    mem0_hist.append(mem_tot[0])
                    mem1_hist.append(mem_tot[1])

            dw0 = jnp.mean(jnp.array(dw0_list), axis=0)
            dw1 = jnp.mean(jnp.array(dw1_list), axis=0)

            dw0_mean_hist.append(dw0)
            dw1_mean_hist.append(dw1)

            if self.save_dw:
                l_dw_hist_h.append(dw0)
                l_dw_hist_o.append(dw1)

            if self.save_spk:
                # print("spk0_hist", jnp.array(spk0_hist).shape)
                l_spk_hist_h.append(spk0_hist)
                l_spk_hist_o.append(spk1_hist)

            if self.save_mem:
                l_mem_hist_h.append(mem0_hist)
                l_mem_hist_o.append(mem1_hist)

            if self.save_w:
                l_w_hist_h.append(w[0])
                l_w_hist_o.append(w[1])

            w_0_new = w[0] + dw0  # jnp.clip(w[0] + dw0, a_min=-100, a_max=100)
            w_1_new = w[1] + dw1  # jnp.clip(w[1] + dw1, a_min=0.01, a_max=100)

            w = [w_0_new, w_1_new]

            l_vRd = jnp.average(
                loss.get_van_Rossum_distance(z_data, spk_tot[1], epsilon, self.nb_steps)
            )

            l_L2 = jnp.average(loss.get_L2_loss(z_data, spk_tot[1]))

            l_loss_hist_vRd.append(float(l_vRd))
            l_loss_hist_L2.append(float(l_L2))

            if self.validate_with_other_method:
                key, subkey = jax.random.split(key)
                (
                    mem_tot_valid,
                    spk_tot_valid,
                    p_tot_valid,
                    mem_tot_nw_valid,
                    key_valid,
                ) = snn.run_2l(
                    x_data,
                    w,
                    subkey,
                    self.nb_steps,
                    self.nb_inputs,
                    self.nb_hidden,
                    self.nb_outputs,
                    self.batch_size,
                    self.tau_mem,
                    self.tau_syn,
                    self.delta_uh,
                    self.delta_uo,
                    self.p0,
                    self.beta_h,
                    self.beta_o,
                    self.theta,
                    self.eps_0,
                    self.dt,
                    self.reset_mode,
                    self.valid_stoch,
                    self.sigm,
                )

                if self.save_spk:
                    # print("spk_tot_valid[0]:", jnp.array(spk_tot_valid[0]).shape)
                    spk0_hist_valid.append(spk_tot_valid[0])
                    spk1_hist_valid.append(spk_tot_valid[1])
                    # print(len(spk0_hist_valid), len(l_spk_hist_h))

                if self.save_spk:
                    # print("spk0_hist_valid", jnp.array(spk0_hist_valid).shape)
                    l_spk_hist_h_valid.append(spk0_hist_valid)
                    l_spk_hist_o_valid.append(spk1_hist_valid)

                l_vRd_valid = jnp.average(
                    loss.get_van_Rossum_distance(
                        z_data, spk_tot_valid[1], epsilon, self.nb_steps
                    )
                )

                l_L2_valid = jnp.average(loss.get_L2_loss(z_data, spk_tot_valid[1]))

                l_loss_hist_vRd_valid.append(float(l_vRd_valid))
                l_loss_hist_L2_valid.append(float(l_L2_valid))

            if (e + 1) % self.logging_freq == 0 or e == self.nb_epochs - 1:
                log.info("Epoch: " + str(e) + ", vRd-loss: " + str(l_vRd))

                #################################################################################################
                # SAVE
                #################################################################################################

                w_dir = os.getcwd()

                loss_hist_vRd = jnp.stack(l_loss_hist_vRd)
                loss_hist_L2 = jnp.stack(l_loss_hist_L2)
                # print(loss_hist_vRd.shape)

                utils.create_and_jnpsave(
                    w_dir, "data/loss_hist_vRd_" + str(e) + ".npy", loss_hist_vRd
                )
                utils.create_and_jnpsave(
                    w_dir, "data/loss_hist_L2_" + str(e) + ".npy", loss_hist_L2
                )
                log.info("saved loss history")

                l_loss_hist_vRd = []
                l_loss_hist_L2 = []

                if self.validate_with_other_method:
                    loss_hist_vRd_valid = jnp.stack(l_loss_hist_vRd_valid)
                    loss_hist_L2_valid = jnp.stack(l_loss_hist_L2_valid)

                    utils.create_and_jnpsave(
                        w_dir,
                        "data/loss_hist_vRd_valid_" + str(e) + ".npy",
                        loss_hist_vRd_valid,
                    )
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/loss_hist_L2_valid_" + str(e) + ".npy",
                        loss_hist_L2_valid,
                    )
                    log.info("saved valid loss history")

                    l_loss_hist_vRd_valid = []
                    l_loss_hist_L2_valid = []

                    if self.save_spk:
                        # print("l_spk_hist_o_valid:", jnp.array(l_spk_hist_o_valid).shape)
                        utils.create_and_jnpsave(
                            w_dir,
                            "data/spk_hist_h_valid_" + str(e) + ".npy",
                            jnp.array(l_spk_hist_h_valid),
                        )
                        utils.create_and_jnpsave(
                            w_dir,
                            "data/spk_hist_o_valid_" + str(e) + ".npy",
                            jnp.array(l_spk_hist_o_valid),
                        )
                        log.info("saved valid spk history")

                        spk0_hist_valid = []
                        spk1_hist_valid = []

                utils.create_and_jnpsave(
                    w_dir, "data/dw0_mean_hist_" + str(e) + ".npy", dw0_mean_hist
                )
                utils.create_and_jnpsave(
                    w_dir, "data/dw1_mean_hist_" + str(e) + ".npy", dw1_mean_hist
                )
                log.info("saved mean weight hist")
                dw0_mean_hist = []
                dw1_mean_hist = []

                if self.save_spk:
                    # print("l_spk_hist_h:", jnp.array(l_spk_hist_h).shape)
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/spk_hist_h_" + str(e) + ".npy",
                        jnp.array(l_spk_hist_h),
                    )
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/spk_hist_o_" + str(e) + ".npy",
                        jnp.array(l_spk_hist_o),
                    )
                    log.info("saved spike hist")
                    l_spk_hist_h = []
                    l_spk_hist_o = []

                if self.save_mem:
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/mem_hist_h_" + str(e) + ".npy",
                        jnp.array(l_mem_hist_h),
                    )
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/mem_hist_o_" + str(e) + ".npy",
                        jnp.array(l_mem_hist_o),
                    )
                    log.info("saved mem hist")
                    l_mem_hist_h = []
                    l_mem_hist_o = []

                if self.save_dw:
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/dw_hist_h_" + str(e) + ".npy",
                        jnp.array(l_dw_hist_h),
                    )
                    utils.create_and_jnpsave(
                        w_dir,
                        "data/dw_hist_o_" + str(e) + ".npy",
                        jnp.array(l_dw_hist_o),
                    )
                    log.info("saved dw hist")
                    l_dw_hist_h = []
                    l_dw_hist_o = []

                if self.save_w:
                    utils.create_and_jnpsave(
                        w_dir, "data/w_hist_h_" + str(e) + ".npy", jnp.array(l_w_hist_h)
                    )
                    utils.create_and_jnpsave(
                        w_dir, "data/w_hist_o_" + str(e) + ".npy", jnp.array(l_w_hist_o)
                    )
                    log.info("saved w hist")
                    l_w_hist_h = []
                    l_w_hist_o = []

                log.info("saved")

        log.info("finished training")

        # return mem_tot, spk_tot, p_tot
