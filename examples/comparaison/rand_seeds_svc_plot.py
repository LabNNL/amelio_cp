import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

path = "examples/results/svc_vs_svr_rdm_state/random_seeds/"


def get_param(dict, param: str):
    return [dict[key][param] for key in dict.keys()]


def categorical_to_numerical(L):
    codes, uniques = pd.factorize(L)
    return codes.tolist(), uniques.tolist()


def plot_scatter(path, type, cond):
    full_path = path + type + "_" + cond + ".pkl"
    with open(full_path, "rb") as file:
        dict = pkl.load(file)

    seed_name = "random_state_" + type
    seeds = get_param(dict, seed_name)
    acc = get_param(dict, "precision_score")
    kernels = get_param(dict, "kernel")
    kernels_num, kernel_names = categorical_to_numerical(kernels)

    mean = np.mean(acc)
    sd = np.std(acc)

    fig = plt.figure()
    plt.scatter(seeds, acc, c=kernels_num, cmap="tab10")
    plt.xlabel("seeds")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])
    plt.text(1, 0.2, f"Mean accuracy: {mean:.2f} ± {sd:.2f}")
    plt.title(f"Accuracy vs. seed\nfor {cond}, varying seed: {type}")

    cmap = cm.get_cmap("tab10")
    norm = mcolors.Normalize(vmin=0, vmax=len(kernel_names) - 1)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=kernel, markerfacecolor=cmap(norm(i)), markersize=8)
        for i, kernel in enumerate(kernel_names)
    ]

    plt.legend(handles=legend_elements, title="Kernel")
    plt.show()


for type in ["split", "cv", "optim"]:
    for cond in ["VIT", "6MWT"]:
        plot_scatter(path, type, cond)
