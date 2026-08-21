import argparse

import pandas as pd

from .train import MAX_RATIO_DRIFT, REFERENCE_POSITIVE_RATIO


def check_positive_class_ratio(data_path: str) -> tuple[float, float, bool]:
    data = pd.read_csv(data_path, usecols=["target"])
    positive_ratio = float(data["target"].mean())
    ratio_drift = abs(positive_ratio - REFERENCE_POSITIVE_RATIO)
    return positive_ratio, ratio_drift, ratio_drift > MAX_RATIO_DRIFT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    args = parser.parse_args()

    positive_ratio, ratio_drift, drift_warning = check_positive_class_ratio(
        args.data_path
    )
    status = "WARNING" if drift_warning else "OK"
    print(
        f"Data drift {status}: positive ratio={positive_ratio:.4f}, "
        f"reference={REFERENCE_POSITIVE_RATIO:.4f}, "
        f"difference={ratio_drift:.4f}, limit={MAX_RATIO_DRIFT:.2f}"
    )


if __name__ == "__main__":
    main()
