import pandas as pd

from src.check_data import check_positive_class_ratio


def test_data_drift_warning_for_large_ratio_change(tmp_path):
    data_path = tmp_path / "drifted.csv"
    pd.DataFrame({"target": [1] * 8 + [0] * 2}).to_csv(data_path, index=False)

    positive_ratio, ratio_drift, warning = check_positive_class_ratio(str(data_path))

    assert positive_ratio == 0.8
    assert ratio_drift > 0.05
    assert warning is True
