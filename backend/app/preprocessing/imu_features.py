import numpy as np
import pandas as pd


IMU_FEATURE_NAMES = [
    "AccMeanX",
    "AccMeanY",
    "AccMeanZ",
    "AccSkewX",
    "AccSkewY",
    "AccSkewZ",
    "AccKurtX",
    "AccKurtY",
    "AccKurtZ",
    "AccMinX",
    "AccMinY",
    "AccMinZ",
    "AccMaxX",
    "AccMaxY",
    "AccMaxZ",
    "AccMedianX",
    "AccMedianY",
    "AccMedianZ",
    "AccStdX",
    "AccStdY",
    "AccStdZ",
    "GyroMeanX",
    "GyroMeanY",
    "GyroMeanZ",
    "GyroSkewX",
    "GyroSkewY",
    "GyroSkewZ",
    "GyroKurtX",
    "GyroKurtY",
    "GyroKurtZ",
    "GyroMinX",
    "GyroMinY",
    "GyroMinZ",
    "GyroMaxX",
    "GyroMaxY",
    "GyroMaxZ",
    "GyroMedianX",
    "GyroMedianY",
    "GyroMedianZ",
    "GyroStdX",
    "GyroStdY",
    "GyroStdZ",
]


def extract_imu_features(samples: list[dict]) -> pd.DataFrame:
    """
    Convert a 4-sample raw IMU window into the exact
    42-feature representation expected by the trained model.

    Expected input keys:
        AccX, AccY, AccZ,
        GyroX, GyroY, GyroZ
    """

    if len(samples) != 4:
        raise ValueError(
            f"IMU model requires exactly 4 samples, received {len(samples)}."
        )

    df = pd.DataFrame(samples)

    required_columns = [
        "AccX",
        "AccY",
        "AccZ",
        "GyroX",
        "GyroY",
        "GyroZ",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing IMU columns: {missing_columns}"
        )

    features = {}

    # Accelerometer features
    for axis in ["X", "Y", "Z"]:
        column = f"Acc{axis}"

        features[f"AccMean{axis}"] = df[column].mean()
        features[f"AccSkew{axis}"] = df[column].skew()
        features[f"AccKurt{axis}"] = df[column].kurt()
        features[f"AccMin{axis}"] = df[column].min()
        features[f"AccMax{axis}"] = df[column].max()
        features[f"AccMedian{axis}"] = df[column].median()
        features[f"AccStd{axis}"] = df[column].std()

    # Gyroscope features
    for axis in ["X", "Y", "Z"]:
        column = f"Gyro{axis}"

        features[f"GyroMean{axis}"] = df[column].mean()
        features[f"GyroSkew{axis}"] = df[column].skew()
        features[f"GyroKurt{axis}"] = df[column].kurt()
        features[f"GyroMin{axis}"] = df[column].min()
        features[f"GyroMax{axis}"] = df[column].max()
        features[f"GyroMedian{axis}"] = df[column].median()
        features[f"GyroStd{axis}"] = df[column].std()

    feature_df = pd.DataFrame([features])

    # Force the exact feature order expected by the model.
    feature_df = feature_df[IMU_FEATURE_NAMES]

    # Ensure all values are numeric.
    feature_df = feature_df.astype(float)

    return feature_df