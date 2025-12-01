from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def plot_AUC(fpr, tpr, auc_value, show=False):
    if show:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_value:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")

        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


def plot_auc_values(auc_values_list_1, auc_values_list_2, condition, model_name, output_path):
    plt.figure(figsize=(20, 7))
    x = np.linspace(1, len(auc_values_list_1), len(auc_values_list_1))
    mean_auc_1 = np.mean(auc_values_list_1)
    std_auc_1 = np.std(auc_values_list_1)
    mean_auc_2 = np.mean(auc_values_list_2)
    std_auc_2 = np.std(auc_values_list_2)

    plt.scatter(x, auc_values_list_1, c="b", label="with ankle")
    plt.scatter(x, auc_values_list_2, c="r", label="without ankle")

    plt.hlines(mean_auc_1, xmin=0, xmax=101, linestyles="--", colors="blue", alpha=0.3)
    plt.annotate(f"Mean ± SD = {mean_auc_1:.3f} ± {std_auc_1:.3f}", xy=(70, 0.15), color="blue", fontsize=12)
    plt.hlines(mean_auc_2, xmin=0, xmax=101, linestyles="--", colors="red", alpha=0.3)
    plt.annotate(f"Mean ± SD = {mean_auc_2:.3f} ± {std_auc_2:.3f}", xy=(70, 0.1), color="red", fontsize=12)

    plt.xlabel("Seeds")
    plt.xticks(x, rotation=90)
    plt.ylabel("AUC values")
    plt.ylim([0, 1.05])
    plt.title(f"AUC values for {condition} with {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        f"{output_path}AUC_plot_for_{condition}_{model_name}_with_without_ankle.svg", dpi=300, bbox_inches="tight"
    )
    plt.show()


def collect_auc_values(dict_model):
    auc_values_list = []
    for key in dict_model.keys():
        y_true = dict_model[key]["y_true"]
        model = dict_model[key]["model"]
        y_score = model.model.decision_function(model.X_test_scaled)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        auc_values_list.append(auc_value)
        plot_AUC(fpr, tpr, auc_value)
    return auc_values_list


def main(condition, model_name, output_path):
    path_svc_19 = f"examples/results/saved_general_data/{model_name}/split_{condition}.pkl"
    with open(path_svc_19, "rb") as file:
        dict_model_19 = pkl.load(file)
    path_svc_19 = f"examples/results/svc_vs_svr_rdm_state/results_without_ankle/{model_name}_{condition}.pkl"
    with open(path_svc_19, "rb") as file:
        dict_model_ss_ankle = pkl.load(file)

    auc_values_19 = collect_auc_values(dict_model_19)
    auc_values_ss_ankle = collect_auc_values(dict_model_ss_ankle)
    plot_auc_values(auc_values_19, auc_values_ss_ankle, condition, model_name, output_path)


if __name__ == "__main__":
    output_path = "examples/results/svc_vs_svr_rdm_state/results_without_ankle/"
    main("VIT", "svc", output_path)
    main("6MWT", "svc", output_path)
