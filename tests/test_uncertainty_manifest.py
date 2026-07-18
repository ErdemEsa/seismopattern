import json
from pathlib import Path
import pytest


def test_uncertainty_manifest_if_present():
    path = Path("output/uncertainty_models/bootstrap_manifest.json")
    if not path.exists():
        pytest.skip("bootstrap manifest repoda yok, test skip")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "n_bootstrap" in data
    assert "models" in data
    for tip in ["TIP_A", "TIP_B", "TIP_C"]:
        assert tip in data["models"], f"{tip} bootstrap modeli eksik"
        assert len(data["models"][tip]) >= 1, f"{tip} icin bootstrap model yok"
