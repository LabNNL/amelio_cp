import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from sklearn.metrics import classification_report
from amelio_cp import SHAPPlots
from amelio_cp import LollipopPlot


def main(file_path, condition_to_predict, feature_names_path, output_path=None):
    model = SVCModel()
    X, y, features_names = Process.prepare_data(file_path, condition_to_predict, model.name, feature_names_path)
    model.add_data(X, y, test_size=0.2)

    # Training the model
    model.train_and_tune("bayesian_optim")
    print(f"Best parameters found for {condition_to_predict} classification model: \n", model.best_params)

    model.dist_from_bound = model.model.decision_function(model.X_test_scaled)
    y_pred_vit_classif = model.model.predict(model.X_test_scaled)
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
    )

    class0_proba = [prob < 0 for prob in model.dist_from_bound]
    class1_proba = [prob > 0 for prob in model.dist_from_bound]
    y_positions = model.y_test.index.tolist()
    class0_ordered, class1_ordered, y_ordered = LollipopPlot.order_values(class0_proba, class1_proba, y_positions)

    LollipopPlot().plot_lollipop(
        class0_ordered,
        class1_ordered,
        y_ordered,
        labels=["Class 0", "Class 1"],
        condition_to_predict=condition_to_predict,
        title=f"Predicted Probabilities for {condition_to_predict}",
        output_path=output_path,
    )

    model.shap_analysis = SHAPPlots.shap_values_calculation(model)
    SHAPPlots.plot_shap_summary(model, features_names, condition_to_predict=condition_to_predict, output_path=output_path, show=True)

if __name__ == "__main__":
    file_path = "datasets/sample_2/all_data_28pp.csv"
    feature_names_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/svc_with_proba/"
    conditions = ["6MWT"]

    for cond in conditions:
        main(file_path, cond, feature_names_path, output_path)