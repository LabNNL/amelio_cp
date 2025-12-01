import numpy as np

from amelio_cp import LinearModel
from amelio_cp import SVRModel
from amelio_cp import ClassifierModel
from amelio_cp import SVCModel
from amelio_cp import Process


data_path = "tests/data/fake_data_for_test.csv"
data = Process.load_csv(data_path)


def test_linear_model():
    model = LinearModel()

    np.testing.assert_almost_equal(
        np.array(
            [
                model.random_state,
                model.params_distributions["C"][0],
                model.params_distributions["C"][1],
                model.params_distributions["gamma"][0],
                model.params_distributions["gamma"][1],
                model.params_distributions["epsilon"][0],
                model.params_distributions["epsilon"][1],
                model.params_distributions["degree"][0],
                model.params_distributions["degree"][1],
                len(model.params_distributions["kernel"]),
            ]
        ),
        np.array([42, 1, 1000, 0.001, 0.1, 0.01, 1, 2, 5, 3]),
    )


def test_classifier_model():
    model = ClassifierModel()

    np.testing.assert_almost_equal(
        np.array(
            [
                model.random_state,
                model.params_distributions["C"][0],
                model.params_distributions["C"][1],
                model.params_distributions["gamma"][0],
                model.params_distributions["gamma"][1],
                model.params_distributions["degree"][0],
                model.params_distributions["degree"][1],
                len(model.params_distributions["kernel"]),
            ]
        ),
        np.array([42, 1, 1000, 0.001, 0.1, 2, 5, 3]),
    )

    np.testing.assert_almost_equal(
        np.array(
            [
                model.random_state,
                model.random_state_split,
                model.random_state_cv,
                model.random_state_optim,
            ]),
            np.array([42, 42, 42, 42]),
    )

    model.random_state = 20

    np.testing.assert_almost_equal(
        np.array(
            [
                model.random_state,
                model.random_state_split,
                model.random_state_cv,
                model.random_state_optim,
            ]),
            np.array([20, 42, 42, 42]),
    )

def test_svr_model():
    model = SVRModel()

    model.best_params = {"C": 50, "gamma": 0.01, "epsilon": 0.1, "degree": 4, "kernel": "rbf"}
    model.model.set_params(**model.best_params)

    np.testing.assert_almost_equal(
        np.array([model.model.C, model.model.gamma, model.model.epsilon, model.model.degree]),
        np.array([50, 0.01, 0.1, 4]),
    )

    X, y, _ = Process.prepare_data(data_path, "VIT", model_name=model.name, features_path=None)
    model.add_data(X, y, test_size=0.2)

    np.testing.assert_equal(np.array(model.X_train.shape), np.array((80, 19)))
    np.testing.assert_equal(
        np.array(model.X_train.index),
        np.array(
            [
                89,
                26,
                42,
                70,
                15,
                40,
                72,
                9,
                96,
                11,
                91,
                64,
                28,
                83,
                5,
                47,
                53,
                35,
                16,
                81,
                34,
                7,
                43,
                73,
                27,
                19,
                94,
                25,
                62,
                49,
                13,
                24,
                3,
                17,
                38,
                8,
                79,
                6,
                65,
                36,
                88,
                56,
                100,
                54,
                50,
                68,
                46,
                69,
                61,
                98,
                80,
                41,
                58,
                48,
                90,
                57,
                75,
                32,
                95,
                59,
                63,
                85,
                37,
                29,
                1,
                52,
                21,
                2,
                23,
                87,
                99,
                74,
                86,
                82,
                20,
                60,
                71,
                14,
                92,
                51,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        model.X_train_scaled[0],
        np.array(
            [-0.81394551, -0.9235252 ,  0.20456801, -0.08627907, -0.10311143,
        0.39241331, -0.1839471 ,  1.30447769,  1.52120961,  0.30404797,
       -0.13384099,  0.83412958, -0.12237037, -1.13388565, -0.31966485,
       -0.27028201, -0.11063276, -1.06845727,  0.66369256]
        ),
    )

    np.testing.assert_almost_equal(
        model.X_test_scaled[10],
        np.array(
            [-0.57346513, -0.11378246,  0.22711145,  0.1438525 , -0.87268322,
       -0.32667434,  2.23228398, -0.16715238, -0.29976348,  1.86015765,
        1.08191249, -0.20763502,  0.14090204, -1.23881797,  0.53998826,
        0.56578258,  1.26804485,  1.91693827, -0.02433134]
        ),
    )
    # TODO: X_test_scaled[i]


def test_svc_model():
    model = SVCModel()

    model.best_params = {"C": 50, "gamma": 0.01, "degree": 4, "kernel": "rbf"}
    model.model.set_params(**model.best_params)

    np.testing.assert_almost_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree]),
        np.array([50, 0.01, 4]),
    )

    X, y, _ = Process.prepare_data(data_path, "VIT", model_name=model.name, features_path=None)
    model.add_data(X, y, test_size=0.2)

    np.testing.assert_equal(np.array(model.X_train.shape), np.array((80, 19)))
    np.testing.assert_equal(
        np.array(model.X_train.index),
        np.array([ 29,  73,  59,  90,  20,   0,  97,  46,  81,  28,  12,  33,  53,  65,
        77,  34,  85,  69,  87,   3,  70,  56,  47,  42,  19,  22,  82,  50,
        13,  71,  14,  27,  72,  84,  54,  15,  26,  60,  58,  36,  68,  48,
        30,  16,  10,  76,   8,  31,  57,  78,  83,  17,  79,  45,  37,  91,
        41,  18,  39,  52,   4,  95,  38,  62, 100,   6,  24,   7,  44,  86,
        89,  49,  66,  74,   5,  75,   9,  51,  11,  43],
        )
    )
    
    np.testing.assert_almost_equal(
        model.X_train_scaled[0],
        np.array(
            [-0.13342089,  0.64625085,  0.93100641, -0.27789055,  0.84442693,
       -0.89071163, -1.44651713, -0.91519077,  1.2878339 ,  1.38228583,
       -0.10834961,  0.30550508,  0.55596304, -1.63477059, -0.62099416,
       -1.11153157,  0.92986218,  0.56831813, -1.3680418 ]
        ),
    )

    np.testing.assert_almost_equal(
        model.y_test.index,
        np.array(
            [88, 67, 21, 25,  2, 55, 40, 96, 63, 93, 35, 61, 98, 64, 99,  1, 92, 80,
       32, 94, 23]
        )
    )

    np.testing.assert_almost_equal(
        model.y_test.to_list(),
        np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
    )