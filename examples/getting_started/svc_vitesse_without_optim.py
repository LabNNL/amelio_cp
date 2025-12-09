import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from sklearn.metrics import classification_report

# %% Collecting/Loading the data from a csv file already created
file_path = "datasets/sample_2/all_data_28pp.csv"

all_data = Process.load_csv(file_path)

# %% Feature selection
features_path = "amelio_cp/processing/Features.xlsx"
features = pd.read_excel(features_path)
selected_features = features["19"].dropna().to_list()
features_names = features["19_names"].dropna().to_list()

# %% Features and labels extraction
X, y, features = Process.prepare_data(file_path, "VIT", 'svc', features_path)

# %% Training the models

SVC_vit = SVCModel()
SVC_vit.random_state = 42
SVC_vit.add_data(X, y, 0.2)
SVC_vit.model.fit(SVC_vit.X_train_scaled, SVC_vit.y_train)

y_pred_vit_classif = SVC_vit.model.predict(SVC_vit.X_test_scaled)
print("Accuracy test score: ", SVC_vit.model.score(SVC_vit.X_test_scaled, SVC_vit.y_test))
print(classification_report(SVC_vit.y_test, y_pred_vit_classif), flush=True)

ClassifierMetrics.conf_matrix(
    SVC_vit,
    SVC_vit.y_test,
    y_pred_vit_classif,
    class_names=["Non-Responder", "Responder"],
    title="Confusion Matrix for speed classification",
)
