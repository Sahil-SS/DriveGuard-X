import math

import pandas as pd

from app.api.schemas import BatteryFeatureVector
from app.services.model_loader import (
    get_battery_model,
    get_battery_schema,
)


def predict_battery_soh(features: BatteryFeatureVector) -> dict:
    model = get_battery_model()
    schema = get_battery_schema()

    if not schema:
        raise RuntimeError("Battery feature schema is missing.")

    values = features.model_dump()
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Battery features must contain only finite numeric values.")

    missing = [name for name in schema if name not in values]
    extra = [name for name in values if name not in schema]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unexpected model features: {extra}"
        )

    row = pd.DataFrame(
        [[values[name] for name in schema]],
        columns=schema,
    )

    prediction = float(model.predict(row)[0])

    if not math.isfinite(prediction):
        raise RuntimeError("Battery model returned a non-finite prediction.")

    return {
        "model": "driveguard_x_final_battery_rf",
        "soh": prediction,
        "soh_percent": prediction * 100.0,
        "input_feature_count": len(schema),
    }
