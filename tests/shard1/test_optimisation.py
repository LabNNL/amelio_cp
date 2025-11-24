import numpy as np

from amelio_cp import LinearModel
from amelio_cp import SVRModel
from amelio_cp import ClassifierModel
from amelio_cp import SVCModel
from amelio_cp import Process

data_path = "tests/data/fake_data_for_test.csv"
data = Process.load_csv(data_path)


def test_optimisation_svc_model():
    model = SVCModel()

    X, y, _ = Process.prepare_data(data_path, features_path, "VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    model.train_and_tune("bayesian_optim", n_iter=20)
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(
        np.array([model.model.C, float(model.model.gamma), model.model.degree]),
        np.array([375.16557872851513, 0.9507635921035062, 4]),
    )

    np.testing.assert_equal(np.array(y_pred), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    model.train_and_tune("bayesian_search", n_iter=20)
    np.testing.assert_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree]),
        np.array([445.3876797888507, 0.011381056696717656, 5]),
    )
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(np.array(y_pred), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
