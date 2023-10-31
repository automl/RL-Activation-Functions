import numpy as np
import time
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.colors as mcolors
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


@hydra.main(version_base=None, config_path="configs", config_name="config_halfcheetah")
def main(cfg: DictConfig):
    t0 = time.time()
    label = ""
    path = "{env}_Plots/".format(env=cfg.ENV_NAME)
    # limit y axis
    lim = 7000
    cfg.ACTIVATION_LIST = "tanh, tanh, tanh, tanh, tanh"
    # activation_list = ["tanh"]
    # activation = ["tanh"]
    # policy_plot(cfg, path, lim)
    # critic_plot(cfg, path, lim)
    # layers_plot(cfg, path, lim)
    layers_policy3_plot(cfg, path, lim)
    # For entropy plots set cfg.ENT_COEF
    # cfg.ENT_COEF = 0.01
    # policy_plot(cfg, path, lim)
    # critic_plot(cfg, path, lim)
    # layers_plot(cfg, path, lim)
    # heatmap_plot(cfg, path)


def moving_avg(arr):
    num_seeds, num_updates, num_steps, num_envs = arr.shape
    window_size = 100
    if num_envs <= 2:
        moving_average = np.zeros((num_seeds, num_updates - window_size + 1))
    else:
        moving_average = np.zeros(
            (num_seeds, num_updates * num_steps - window_size + 1))
    for i in range(num_seeds):
        if num_envs <= 2:
            data = arr[i].mean((-2, -1))
        else:
            data = arr[i].mean(-1).reshape(-1)

        moving_average_value = np.convolve(
            data, np.ones(window_size)/window_size, 'valid')
        moving_average[i] = moving_average_value
    return moving_average


def heatmap_plot(cfg, path):
    def shape_arr(arr, steps_percent):
        num_seeds, num_updates, num_steps, num_envs, pos = arr.shape
        steps = int(steps_percent * num_updates)
        arr_2d = np.zeros((13, 13), dtype=int)
        arr_sub = arr[:, :steps, :, :, :2]
        flat_arr = arr_sub.reshape(-1, 2)
        unique_entries, counts = np.unique(
            flat_arr, axis=0, return_counts=True)
        unique_entries_with_counts = np.column_stack((unique_entries, counts))
        for i in range(unique_entries_with_counts.shape[0]):
            x = unique_entries_with_counts[i][0]
            y = unique_entries_with_counts[i][1]
            arr_2d[y, x] = unique_entries_with_counts[i][2]
        return arr_2d

    if (cfg.LAYER == True):
        activation_list = cfg.ACTIVATION_LIST.split(", ")
    else:
        activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")

    activation = cfg.ACTIVATION_FUNCTIONS.split(", ")
    # activation_list = ["tanh"]
    percent = 0.25
    activation = ["tanh"]
    # Swap af_policy and af_critic if necessary, if plotting Layers set cfg.LAYER=True and make sure order of af_critic and af_policy is the same as in the training
    for j, af_critic in enumerate(activation_list):
        for af_policy in activation:
            runs = int(1/percent)
            for i in range(runs):
                if (cfg.LAYER == True and cfg.ENT_COEF != 0):
                    label = "{env}_entCoef_Policy_{policy}_Critic_{critic}_Layer{num}".format(env=cfg.ENV_NAME,
                                                                                              policy=af_policy, critic=af_critic, num=j+1)
                elif (cfg.LAYER == True):
                    label = "{env}_Policy_{policy}_Critic_{critic}_Layer{num}".format(env=cfg.ENV_NAME,
                                                                                      policy=af_policy, critic=af_critic, num=j+1)
                elif (cfg.ENT_COEF != 0):
                    label = "{env}_entCoef_Policy_{policy}_Critic_{critic}".format(env=cfg.ENV_NAME,
                                                                                   policy=af_policy, critic=af_critic)
                else:
                    label = "{env}_Policy_{policy}_Critic_{critic}".format(env=cfg.ENV_NAME,
                                                                           policy=af_policy, critic=af_critic)
                wall_pos = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12),
                            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7,
                                                                             0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
                            (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12,
                                                                                   7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12),
                            (1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6,
                                                                          12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12),
                            (1, 6), (3, 6), (4, 6), (4, 6), (5, 6), (6,
                                                                     6), (7, 7), (8, 7), (10, 7), (11, 7),
                            (6, 1), (6, 2), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (6, 11)]
                shaped_arr = np.load(path+"numpy/"+label+".npy")
                shaped_arr = shape_arr(shaped_arr, percent*(i+1))
                min_val = np.min(shaped_arr)
                max_val = np.max(shaped_arr)
                shaped_arr = 100 * \
                    (shaped_arr - min_val) / (max_val - min_val)
                mask = np.zeros(shaped_arr.shape, dtype=bool)
                for position in wall_pos:
                    mask[position] = True
                masked_data = np.ma.masked_array(shaped_arr, mask)

                colors = [(1, 1, 1), (0, 0.5, 0)]  # White to Dark Green
                cmap_name = 'white_darkgreen'
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    cmap_name, colors, N=256)
                cmap.set_bad(color="black")

                cords = np.array([4, 1, 8, 9])
                title = "Heatmap for Fourrooms state visits"
                plt.figure(figsize=(10, 6))
                plt.text(cords[0], cords[1], "Start",
                         horizontalalignment="center", fontweight="bold", color="red")
                plt.text(cords[2], cords[3], "Goal",
                         horizontalalignment="center", fontweight="bold", color="red")
                plt.imshow(masked_data, cmap=cmap,
                           interpolation='nearest', vmin=0, vmax=100)
                plt.colorbar(format="%.0f%%", ticks=[0, 100])

                plt.title(title)
                label = "{run}_".format(run=int(percent*100*(i+1))) + label
                plt.savefig(path+label)
                plt.close()


def layers_policy3_plot(cfg, path, lim):
    activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    layer = True
    if (layer == True):
        activation_list = cfg.ACTIVATION_LIST.split(", ")
    else:
        activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    activation = cfg.ACTIVATION_FUNCTIONS.split(", ")
    for j, af_critic in enumerate(activation_list):
        IQM_values_list = []
        if cfg.ENT_COEF != 0:
            IQM_values_list = np.load(
                path + "numpy/{env}_entcoefPolicy3_Layer{num}.npy".format(env=cfg.ENV_NAME, af=af_critic, num=j+1))

        else:
            IQM_values_list = np.load(
                path + "numpy/{env}_Policy3_Layer{num}.npy".format(env=cfg.ENV_NAME, af=af_critic, num=j+1))
        plt.figure(figsize=(10, 6))
        plt.ylim(0, lim)
        for i, af_policy in enumerate(activation):
            std = np.std(IQM_values_list[i], axis=0)
            num_steps = IQM_values_list[i].shape[1]
            IQM_values = np.array([metrics.aggregate_iqm(IQM_values_list[i][:, t])
                                   for t in range(num_steps)])
            lower_bound = IQM_values[:num_steps] - std
            upper_bound = IQM_values[:num_steps] + std
            label = "AF: {policy}".format(policy=af_policy)
            plt.plot(range(num_steps),
                     IQM_values[:num_steps], label=label)
            plt.fill_between(range(num_steps),
                             lower_bound, upper_bound, alpha=0.2)

        title = "IQM returns for fixed AF: {af} and separate AF in a Layer".format(
            af=af_critic)
        if cfg.ENT_COEF != 0:
            file_name = "{env}_entCoef_Policy3_Layer_{num}_{af}".format(
                env=cfg.ENV_NAME, af=af_critic, num=j)

        else:
            file_name = "{env}_Policy3_Layer_{num}_{af}".format(
                env=cfg.ENV_NAME, af=af_critic, num=j)
        # np.save(file_name, IQM_values_list)

        plt.xlabel("Number of Steps")
        plt.ylabel("Returns")
        plt.title(title)
        plt.legend()
        file_name = path+file_name
        plt.savefig(file_name)
        plt.close()


def layers_plot(cfg, path, lim):
    activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    layer = True
    if (layer == True):
        activation_list = cfg.ACTIVATION_LIST.split(", ")
    else:
        activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    activation = cfg.ACTIVATION_FUNCTIONS.split(", ")
    for j, af_critic in enumerate(activation_list):
        IQM_values_list = []
        if cfg.ENT_COEF != 0:
            IQM_values_list = np.load(
                path + "numpy/{env}_entCoef_Critic_{af}_Layer{num}.npy".format(env=cfg.ENV_NAME, af=af_critic, num=j+1))

        else:
            IQM_values_list = np.load(
                path + "numpy/{env}_Critic_{af}_Layer{num}.npy".format(env=cfg.ENV_NAME, af=af_critic, num=j+1))
        plt.figure(figsize=(10, 6))
        plt.ylim(0, lim)
        for i, af_policy in enumerate(activation):
            std = np.std(IQM_values_list[i], axis=0)
            num_steps = IQM_values_list[i].shape[1]
            IQM_values = np.array([metrics.aggregate_iqm(IQM_values_list[i][:, t])
                                   for t in range(num_steps)])
            lower_bound = IQM_values[:num_steps] - std
            upper_bound = IQM_values[:num_steps] + std
            label = "AF: {policy}".format(policy=af_policy)
            plt.plot(range(num_steps),
                     IQM_values[:num_steps], label=label)
            plt.fill_between(range(num_steps),
                             lower_bound, upper_bound, alpha=0.2)

        title = "IQM returns for fixed AF: {af} and separate AF in a Layer".format(
            af=af_critic)
        if cfg.ENT_COEF != 0:
            file_name = "{env}_entCoef_Layer_{num}_{af}".format(
                env=cfg.ENV_NAME, af=af_critic, num=j)

        else:
            file_name = "{env}_Layer_{num}_{af}".format(
                env=cfg.ENV_NAME, af=af_critic, num=j)
        # np.save(file_name, IQM_values_list)

        plt.xlabel("Number of Steps")
        plt.ylabel("Returns")
        plt.title(title)
        plt.legend()
        file_name = path+file_name
        plt.savefig(file_name)
        plt.close()


def critic_plot(cfg, path, lim):
    activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    for j, af_critic in enumerate(activation_list):
        IQM_values_list = []
        if (cfg.ENT_COEF != 0):
            IQM_values_list = np.load(
                path+"numpy/{env}_entCoef_Critic_{af}.npy".format(env=cfg.ENV_NAME, af=af_critic))
        else:
            IQM_values_list = np.load(
                path+"numpy/{env}_Critic_{af}.npy".format(env=cfg.ENV_NAME, af=af_critic))
        plt.figure(figsize=(10, 6))
        plt.ylim(0, lim)

        for i, af_policy in enumerate(activation_list):
            std = np.std(IQM_values_list[i], axis=0)
            num_steps = IQM_values_list[i].shape[1]
            IQM_values = np.array([metrics.aggregate_iqm(IQM_values_list[i][:, t])
                                   for t in range(num_steps)])
            lower_bound = IQM_values[:num_steps] - std
            upper_bound = IQM_values[:num_steps] + std
            label = "Policy: {policy}".format(policy=af_policy)
            plt.plot(range(num_steps),
                     IQM_values[:num_steps], label=label)
            plt.fill_between(range(num_steps),
                             lower_bound, upper_bound, alpha=0.2)

        if (cfg.ENT_COEF != 0):
            title = "IQM returns for critic {af}".format(af=af_critic)
            file_name = "{env}_entCoef_Critic_{af}".format(
                env=cfg.ENV_NAME, af=af_critic)
        else:
            title = "IQM returns for critic {af}".format(af=af_critic)
            file_name = "{env}_Critic_{af}".format(
                env=cfg.ENV_NAME, af=af_critic)
        # np.save(file_name, IQM_values_list)

        plt.xlabel("Number of Steps")
        plt.ylabel("Returns")
        plt.title(title)
        plt.legend()
        file_name = path+file_name
        plt.savefig(file_name)
        plt.close()


def policy_plot(cfg, path, lim):
    activation_list = cfg.ACTIVATION_FUNCTIONS.split(", ")
    for i, af_policy in enumerate(activation_list):
        IQM_values_list = []
        plt.figure(figsize=(10, 6))
        plt.ylim(0, lim)

        for j, af_critic in enumerate(activation_list):
            if (cfg.ENT_COEF != 0):
                IQM_values_list = np.load(
                    path+"numpy/{env}_entCoef_Critic_{af}.npy".format(env=cfg.ENV_NAME, af=af_critic))
            else:
                IQM_values_list = np.load(
                    path+"numpy/{env}_Critic_{af}.npy".format(env=cfg.ENV_NAME, af=af_critic))

            std = np.std(IQM_values_list[i], axis=0)
            num_steps = IQM_values_list[i].shape[1]
            IQM_values = np.array([metrics.aggregate_iqm(IQM_values_list[i][:, t])
                                   for t in range(num_steps)])
            lower_bound = IQM_values[:num_steps] - std
            upper_bound = IQM_values[:num_steps] + std
            label = "Critic: {policy}".format(policy=af_critic)
            plt.plot(range(num_steps),
                     IQM_values[:num_steps], label=label)
            plt.fill_between(range(num_steps),
                             lower_bound, upper_bound, alpha=0.2)

        if (cfg.ENT_COEF != 0):
            title = "IQM returns for policy {af}".format(af=af_policy)
            file_name = "{env}_entCoef_Policy_{af}".format(
                env=cfg.ENV_NAME, af=af_policy)
        else:
            title = "IQM returns for policy {af}".format(af=af_policy)
            file_name = "{env}_Policy_{af}".format(
                env=cfg.ENV_NAME, af=af_policy)
        # np.save(file_name, IQM_values_list)

        plt.xlabel("Number of Steps")
        plt.ylabel("Returns")
        plt.title(title)
        plt.legend()
        file_name = path+file_name
        plt.savefig(file_name)
        plt.close()


if __name__ == "__main__":
    main()
