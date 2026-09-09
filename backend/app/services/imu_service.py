from app.preprocessing.imu_features import extract_imu_features
from app.services.model_loader import get_imu_model


def predict_imu(request):
    """
    Run IMU driving-behavior inference.

    Input:
        Exactly 4 raw IMU samples.

    Processing:
        4 raw samples
            ↓
        42 statistical features
            ↓
        XGBoost
            ↓
        DriveGuard-X class 1-4
    """

    model = get_imu_model()

    samples = [
        sample.model_dump()
        for sample in request.samples
    ]

    feature_df = extract_imu_features(samples)

    prediction = model.predict(feature_df)

    # XGBoost uses zero-based class indices internally.
    # DriveGuard-X uses classes 1-4.
    predicted_class = int(prediction[0]) + 1

    return {
        "model": "driveguard_xgb_final",
        "predicted_class": predicted_class,
        "feature_count": len(feature_df.columns),
        "window_size": len(samples),
    }