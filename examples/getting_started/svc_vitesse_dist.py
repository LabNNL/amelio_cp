import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from sklearn.metrics import classification_report
from amelio_cp import SHAPPlots
from amelio_cp import LollipopPlot
import numpy as np


def main(file_path, condition_to_predict, feature_names_path, output_path=None, show=True):
    model = SVCModel()
    X, y, features_names = Process.prepare_data(file_path, condition_to_predict, model.name, feature_names_path)
    if condition_to_predict == "VIT":
        X = X.drop([50, 51], axis=0)
        y = y.drop([50, 51], axis=0)

    model.add_data(X, y, test_size=0.2)

    # Manually setting train and test data to have the same samples for comparison
    # model.X_test = X.loc[[39, 49, 10, 54, 1, 33, 6, 53, 31, 5, 30]]
    # model.y_test = y.loc[[39, 49, 10, 54, 1, 33, 6, 53, 31, 5, 30]]
    # model.X_train = X.drop([39, 49, 10, 54, 1, 33, 6, 53, 31, 5, 30], axis=0)
    # model.y_train = y.drop([39, 49, 10, 54, 1, 33, 6, 53, 31, 5, 30], axis=0)

    # model.X_train_scaled = model.scaler.fit_transform(model.X_train)
    # model.X_test_scaled = model.scaler.transform(model.X_test)

    # Training the model
    model.train_and_tune("bayesian_optim")
    print(f"Best parameters found for {condition_to_predict} classification model: \n", model.best_params)

    model.dist_from_bound = model.model.decision_function(model.X_test_scaled)
    y_pred_vit_classif = model.model.predict(model.X_test_scaled)
    print("\nTrue labels: ", model.y_test)
    print("\nPredictions: ", y_pred_vit_classif)
    print("Accuracy test score: ", model.model.score(model.X_test_scaled, model.y_test))
    print(classification_report(model.y_test, y_pred_vit_classif), flush=True)

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

    y_positions = model.y_test.index.tolist()
    responders_mask = model.y_test.to_numpy() == y_pred_vit_classif

    LollipopPlot().plot_lollipop_decision(
        x=model.dist_from_bound,
        y=y_positions,
        responders_mask=responders_mask,
        labels=["Non-Responders", "Responders"],
        condition_to_predict=condition_to_predict,
        title=f"Distances from the decision boundary for {condition_to_predict}",
        output_path=output_path,
        show=show,
    )

    model.shap_analysis = SHAPPlots.shap_values_calculation(model)
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
