from pathlib import Path


def test_readme_exists():
    assert Path("README.md").exists(), "README.md bulunamadi"


def test_docs_files_exist():
    required = [
        "docs/Limitations.md",
        "docs/Scientific_Methodology.md",
        "docs/Validation_Methodology.md",
        "docs/Feature_Definitions.md",
        "docs/Risk_Model.md",
        "docs/Calibration.md",
        "docs/DEPLOYMENT.md",
    ]
    for p in required:
        assert Path(p).exists(), f"Eksik belge: {p}"


def test_readme_has_core_terms():
    txt = Path("README.md").read_text(encoding="utf-8").lower()
    assert "seismopattern" in txt
    assert "prospective" in txt
    assert "docker" in txt or "compose" in txt


def test_deployment_has_runtime_compose():
    txt = Path("docs/DEPLOYMENT.md").read_text(encoding="utf-8").lower()
    assert "docker compose -f docker-compose.runtime.yml up" in txt
