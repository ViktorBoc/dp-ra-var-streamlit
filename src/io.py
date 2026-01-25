from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def _project_root() -> Path:
    # app.py sits in project_root; src/ is sibling
    return Path(__file__).resolve().parents[1]


def resolve_path(rel_path: str) -> Path:
    """Resolve a path from project_root, with a fallback to /mnt/data/<filename> for sandbox runs."""
    root = _project_root()
    p = root / rel_path
    if p.exists():
        return p

    # Sandbox fallback: files may be mounted flat under /mnt/data
    fname = Path(rel_path).name
    alt = Path("/mnt/data") / fname
    if alt.exists():
        return alt

    raise FileNotFoundError(f"File not found: {p} (also tried {alt})")


def load_csv(rel_path: str, **kwargs) -> pd.DataFrame:
    p = resolve_path(rel_path)
    return pd.read_csv(p, **kwargs)


def load_json(rel_path: str) -> Any:
    p = resolve_path(rel_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yml(rel_path: str) -> Any:
    p = resolve_path(rel_path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_inputs() -> Dict[str, Any]:
    """Load all required local inputs."""
    data: Dict[str, Any] = {}

    data["portfolio"] = load_csv("data/portfolio/dummy_data.csv")
    data["mortality"] = load_csv("data/assumptions/mortality.csv")
    data["lapse"] = load_csv("data/assumptions/lapse_rates.csv")
    data["expenses"] = load_json("data/assumptions/expenses.json")
    data["risk_free"] = load_csv("data/assumptions/risk_free_curve.csv")
    data["inflation"] = load_csv("data/assumptions/inflation_curve_base_and_stressed_ecb.csv")
    data["risk_shocks"] = load_json("data/risk_inputs/risk_shocks_995.json")
    data["corr"] = load_csv("data/risk_inputs/correlation_matrix.csv", index_col=0)
    data["product_risk_map"] = load_yml("data/risk_inputs/product_risk_map.yml")
    data["var_levels"] = load_json("data/config/var_levels.json")
    data["var_scaling"] = load_yml("data/config/var_scaling.yml")

    return data
