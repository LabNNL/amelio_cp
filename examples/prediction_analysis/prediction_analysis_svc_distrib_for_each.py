from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from amelio_cp import Distributions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl


def build_model(rdm_state):
    model = SVCModel()
    model.random_state_split = rdm_state
    return model


def prepare_data(data_path, condition_to_predict, features_path):
    all_data = Process.load_csv(data_path)
    if condition_to_predict == "VIT":
        all_data = all_data.drop(["6MWT_POST"], axis=1)
        all_data = all_data.dropna()
        y = Process.calculate_MCID(all_data["VIT_PRE"], all_data["VIT_POST"], "VIT")

    elif condition_to_predict == "6MWT":
        all_data = all_data.drop(["VIT_POST"], axis=1)
        all_data = all_data.dropna()
        y = Process.calculate_MCID(all_data["6MWT_PRE"], all_data["6MWT_POST"], "6MWT", all_data["GMFCS"])

    else:
        raise ValueError("Condition to predict not recognized. Choose either 'VIT' or '6MWT'.")

    features = pd.read_excel(features_path)
    selected_features = features["19"].dropna().to_list()

    X = all_data[selected_features]

    return X, y, selected_features


def load_data(X, y, model, test_size=0.2):
    model.add_data(X, y, test_size)


def collect_wrong_predictions(y_1, y_2, idx):
    FP, FN = [], []

    for i in range(len(y_1)):
        if y_1[i] - y_2[i] > 0:
            FN.append(idx[i])
        elif y_1[i] - y_2[i] < 0:
            FP.append(idx[i])
        else:
            continue
    return FP, FN


def plot_distributions(data, wrong_preds, features, condition_to_predict, random_state, output_path):
    for i in range(len(wrong_preds)):
        fig, axes = plt.subplots(5, 4, figsize=(50, 50))
        m, n = 0, 0
        for feature in features:
            Distributions.plot_violin(data[feature], ax=axes[m, n], highlight_idx=wrong_preds[i], show=False)
            n += 1
            if n == 4:
                n = 0
                m += 1
        plt.suptitle(f"Distributions for patient index: {wrong_preds[i]}\n(random_state={random_state})", fontsize=40)
        plt.savefig(
            f"{output_path}distribution_for_{wrong_preds[i]}_{condition_to_predict}_{random_state}.svg",
            dpi=300,
            bbox_inches="tight",
        )


def append_data(results_dict, condition_to_predict, model, accuracy, y_true, y_pred, FP, FN):
    results_dict[model.random_state_split] = {
        "condition_to_predict": condition_to_predict,
        "optim_method": model.optim_method,
        "C": model.best_params["C"],
        "gamma": model.best_params["gamma"],
        "degree": model.best_params["degree"],
        "kernel": model.best_params["kernel"],
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "wrong_predictions": {"FP": FP, "FN": FN},
        "idx_test": model.y_test.index.tolist(),
        # "shap_values": model.shap_analysis["shap_values"],
    }

    return results_dict


def save_data(results_dict, condition_to_predict, output_path):
    pickle_file_name = output_path + condition_to_predict + ".pkl"
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)

 
def main(condition_to_predict, list_random_state):
    results_dict = {}
    data_path = "datasets/sample_2/all_data_28pp.csv"
    features_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/prediction_analysis_results/"

    for random_state in list_random_state:
        X, y, features = prepare_data(data_path, condition_to_predict, features_path)
        model = build_model(random_state)
        load_data(X, y, model)

        model.train_and_tune("bayesian_optim")
        y_pred = model.model.predict(model.X_test_scaled)
        accuracy = model.model.score(model.X_test_scaled, model.y_test)

        ClassifierMetrics.conf_matrix(
            model,
            model.y_test,
            y_pred,
            ["Responders", "Non-Responders"],
            "Confusion_Matrix",
        )

        FP, FN = collect_wrong_predictions(np.array(model.y_test), y_pred, model.y_test.index)
        plot_distributions(X, FP, features, condition_to_predict, random_state, output_path + "false_positives_")
        plot_distributions(X, FN, features, condition_to_predict, random_state, output_path + "false_negatives_")

        append_data(results_dict, condition_to_predict, model, accuracy, model.y_test, y_pred, FP, FN)

    save_data(results_dict, condition_to_predict, output_path)


if __name__ == "__main__":
    list_random_state = [42, 72, 36, 8, 17]
    main("VIT", list_random_state)
    # main("6MWT")
