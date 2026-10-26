"""Predefined hyperparameter configurations for model evaluation and comparison."""

PRESETS = {
    "baseline": {
        "penalizer": 0.1,
        "prune_frac": 0.456,
        "smote_k": 5,
        "max_ci_width": None,
        "thresholds": [240, 270, 300, 330, 360],
        "description": "Original baseline configuration (Jamalian CS229 / teammate implementation)",
    },
    "optimal": {
        "penalizer": 0.05,
        "prune_frac": 0.456,
        "smote_k": 5,
        "max_ci_width": None,
        "thresholds": [240, 270, 300, 330, 360],
        "description": "Tuned hyperparameter configuration (optimal L2 penalty, improved honest C-index)",
    },
}


def get_preset(name):
    """Retrieve preset config dictionary by name ('baseline' or 'optimal')."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return dict(PRESETS[name])
