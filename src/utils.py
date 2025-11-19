"""Utility functions for the DPO training pipeline."""

import json
import yaml
from pathlib import Path


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path, indent=2):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_yaml(file_path):
    """Load YAML configuration file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
