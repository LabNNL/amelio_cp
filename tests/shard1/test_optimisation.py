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

    X, y, _ = Process.prepare_data(data_path, "VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    # Bayesian Optimization
    model.train_and_tune("bayesian_optim", n_iter=30)
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(
        np.array([model.model.C, float(model.model.gamma), model.model.degree]),
        np.array([832.6101981596213, 0.21312677156759788, 2]),
    )

    np.testing.assert_equal(np.array(y_pred), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]))

    # Bayesian Search
    model.train_and_tune("bayesian_search", n_iter=30)
    np.testing.assert_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree]),
        np.array([837.5509671977311, 0.031037600878533328, 5]),
    )
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(np.array(y_pred), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]))
    
    # Random Search
    model.train_and_tune("random", n_iter=30)
    np.testing.assert_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree]),
        np.array([525.7564316322379, 0.030122914019804194, 2]),
    )
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(np.array(y_pred), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]))
