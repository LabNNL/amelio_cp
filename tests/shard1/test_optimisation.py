import numpy as np
from amelio_cp import SVRModel
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


def test_optimisation_svr_model():
    model = SVRModel()

    X, y, _ = Process.prepare_data(data_path, "VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    # Bayesian Optimization
    model.train_and_tune("bayesian_optim", n_iter=30)
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(
        np.array([model.model.C, float(model.model.gamma), model.model.epsilon, model.model.degree]),
        np.array([608.2331813928909, 1.0, 1.0, 3]),
    )

    np.testing.assert_equal(np.array(y_pred), np.array([-8.45368369, -8.45368324, -8.45354466, -8.45368683, -8.45369076,
       -8.45368369, -8.45368369, -8.45368369, -8.45370041, -8.45409376,
       -8.4536837 , -8.45368369, -8.4536848 , -8.45368369, -8.45368575,
       -8.45368369, -8.45368369, -8.4536938 , -8.45368369, -8.45368371,
       -8.45368372]))

    # Bayesian Search
    model.train_and_tune("bayesian_search", n_iter=30)
    np.testing.assert_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree, model.model.epsilon]),
        np.array([2.884665005293273, 0.031548608996857345, 4, 0.7690483295833969]),
    )
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(np.array(y_pred), np.array([-8.59839324, -8.70245041, -8.72761332, -8.81022244, -8.76722762,
       -8.69544455, -9.45807104, -8.92287984, -8.79823648, -8.78091242,
       -8.75908081, -8.82061761, -8.74106101, -8.67059022, -8.78793948,
       -9.08981267, -8.76220811, -8.75989761, -8.58894178, -8.93338925,
       -8.75084478])
    )
    
    # Random Search
    model.train_and_tune("random", n_iter=30)
    np.testing.assert_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree, model.model.epsilon]),
        np.array([476.3702231821118, 0.0917566473926093, 2, 0.5185706911647028]),
    )
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(np.array(y_pred), np.array([ -8.6366475 ,  -7.41477441,  -7.41440192,  -8.10812452,
        -8.78241466,  -7.67491361,  -9.2575797 ,  -8.99108447,
        -8.87712118,  -8.33675214,  -7.66509022,  -7.57647913,
        -7.77474909,  -8.09392835,  -9.03658201,  -8.38609765,
        -7.70598998, -10.11615516,  -8.57155189,  -8.3943591 ,
        -8.97915061]))