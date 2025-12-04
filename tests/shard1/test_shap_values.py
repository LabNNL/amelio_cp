import numpy as np
from amelio_cp import SVCModel
from amelio_cp import Process
from amelio_cp import SHAPPlots

data_path = "tests/data/fake_data_for_test.csv"
data = Process.load_csv(data_path)


def test_shap_values():
    model = SVCModel()

    X, y, _ = Process.prepare_data(data_path, "VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    model.model.fit(model.X_train_scaled, model.y_train)
    shap = SHAPPlots.shap_values_calculation(model)
    shap_values = shap["shap_values"]

    np.testing.assert_equal(
        np.array(shap_values[0]),
        np.array([0.00143326, 0.00061773, 0.00135378, 0.00044824, 0.00208691, 0.0,
 0.00186164, 0.00570884, 0.0054981,  0.00019372, 0.0003502,  0.00494942,
 0.00072281, 0.00221921, 0.00282439, 0.00036093, 0.00251335, 0.00404454,
 0.00031293]),
    )
