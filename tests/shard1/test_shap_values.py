import numpy as np
from amelio_cp import SVCModel
from amelio_cp import Process
from amelio_cp import SHAPPlots

data_path = "tests/data/fake_data_for_test.csv"
data = Process.load_csv(data_path)


def test_shap_values():
    model = SVCModel()

    X, y = Process.prepare_data2(data_path=data_path, condition_to_predict="VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    model.model.fit(model.X_train_scaled, model.y_train)
    shap = SHAPPlots.shap_values_calculation(model)
    shap_values = shap["shap_values"]

    np.testing.assert_almost_equal(
        np.array(shap_values[0]),
        np.array(
            [
                0.01194511,
                0.00711466,
                0.01015479,
                0.00894768,
                0.03544341,
                0.00330224,
                0.00375394,
                0.08201682,
                0.0645,
                0.0,
                0.00082611,
                0.00713621,
                0.01868982,
                0.10811215,
                0.0220972,
                0.00578715,
                -0.00626067,
                0.09188869,
                0.0245447,
            ]
        ),
    )
