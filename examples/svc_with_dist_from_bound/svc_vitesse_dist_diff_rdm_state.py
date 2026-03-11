import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from sklearn.metrics import classification_report, confusion_matrix
from amelio_cp import SHAPPlots
from amelio_cp import LollipopPlot
import pickle as pkl
import time


def prepare_data(file_path, condition_to_predict, model_name, feature_names_path):
    X, y, features_names = Process.prepare_data(file_path, condition_to_predict, model_name, feature_names_path)
    if condition_to_predict == "VIT":
        X = X.drop([50, 51], axis=0)
        y = y.drop([50, 51], axis=0)
    return X, y, features_names


def add_data_to_model(
    model,
    X,
    y,
    test_idx: list,
):

    # model.add_data(X, y, test_size=0.2)
    model.X_test = X.loc[test_idx]
    model.y_test = y.loc[test_idx]
    model.X_train = X.drop(test_idx, axis=0)
    model.y_train = y.drop(test_idx, axis=0)

    model.X_train_scaled = model.scaler.fit_transform(model.X_train)
    model.X_test_scaled = model.scaler.transform(model.X_test)

    return model


def append_data(results_dict, model, id, time, model_score, conf_matrix, y_true, y_pred):
    results_dict["id_" + str(id)] = {
        "model_name": model.name,
        "model": model,
        "seed": model.random_state,
        "model_score": model_score,
        "confusion_matrix": conf_matrix,
        "optim_time": time,
        "y_true": y_true,
        "y_pred": y_pred,
        "dist_from_boundary": model.dist_from_bound,
    }

    return results_dict


def save_data(results_dict, condition_to_predict, random_state_optim, output_path):
    pickle_file_name = output_path + random_state_optim + "_" + condition_to_predict + ".pkl"
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def main(file_path, condition_to_predict, feature_names_path, test_idx, output_path=None, show=True):
    result_dict = {}
    model = SVCModel()
    X, y, feature_names = prepare_data(file_path, condition_to_predict, feature_names_path)
    model = add_data_to_model(X, y, test_idx)

    # Train the model
    starting_time = time.time()
    model.train_and_tune("bayesian_optim")
    print(f"Best parameters found for {condition_to_predict} classification model: \n", model.best_params)

    # Collecte distances from the boundary
    model.dist_from_bound = model.model.decision_function(model.X_test_scaled)
    # Calculate the predicted classification
    y_pred_vit_classif = model.model.predict(model.X_test_scaled)

    # Printing the true and predicted classifications (for manual comparison)
    print("\nTrue labels: ", model.y_test)
    print("\nPredictions: ", y_pred_vit_classif)

    # Calculate the model score
    model_score = model.model.score(model.X_test_scaled, model.y_test)
    print("Accuracy test score: ", model_score)
    # Display the metrics
    print(classification_report(model.y_test, y_pred_vit_classif), flush=True)
    # Display the confusion matrix
    confusion_mat = confusion_matrix(model.y_test, y_pred_vit_classif)
    print(f"Confusion Matrix for:\n", confusion_mat)

    # Calculate the shap values
    model.shap_analysis = SHAPPlots.shap_values_calculation(model)

    # End the time of the trial
    simulation_time = time.time() - starting_time

    # Derive the mask of the correctly/wrongly predicted
    responders_mask = model.y_test.to_numpy() == y_pred_vit_classif

    result_dict = append_data(
        result_dict, model, id, simulation_time, model_score, confusion_matrix, model.y_test, y_pred_vit_classif
    )

    ClassifierMetrics.conf_matrix(
        model,
        model.y_test,
        y_pred_vit_classif,
        class_names=["Non-Responder", "Responder"],
        condition_to_predict=condition_to_predict,
        title=f"Confusion Matrix for {condition_to_predict} classification",
        output_path=output_path,
        show=show,
    )

    LollipopPlot().plot_lollipop_decision(
        x=model.dist_from_bound,
        y=model.y_test.index.tolist(),
        responders_mask=responders_mask,
        labels=["Non-Responders", "Responders"],
        condition_to_predict=condition_to_predict,
        title=f"Distances from the decision boundary for {condition_to_predict}",
        output_path=output_path,
        show=show,
    )

    SHAPPlots.plot_shap_summary(
        model, features_names, condition_to_predict=condition_to_predict, output_path=output_path, show=show
    )


if __name__ == "__main__":
    file_path = "datasets/sample_2/all_data_28pp.csv"
    feature_names_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/svc_with_dist/same_samples/"
    conditions = ["VIT", "6MWT"]
    show = True

    for cond in conditions:
        print(f"\n\n===== Processing condition: {cond} =====\n")
        main(file_path, cond, feature_names_path, show)
