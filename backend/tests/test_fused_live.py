from app.services.attribution_service import attribution_engine


def test_fused_live_result_preserves_attribution_and_protection_fields():
    result = attribution_engine.fuse_results(
        {"soh_percent": 72.0, "stressed": True},
        {"predicted_class": 1},
        {"road_class": "smooth", "confidence": 0.9},
    )
    assert result["attribution"]["likely_cause"] == "battery_degraded"
    assert result["protection"]["pwm_limit_pct"] == 65


def test_fused_live_result_is_normal_when_battery_is_not_stressed():
    result = attribution_engine.fuse_results(
        {"soh_percent": 96.0, "stressed": False},
        {"predicted_class": 2},
        {"road_class": "rough", "confidence": 0.9},
    )
    assert result["attribution"]["likely_cause"] == "normal"
    assert result["protection"]["pwm_limit_pct"] == 100
