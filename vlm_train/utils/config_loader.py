import yaml
import os

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        # Fallback to root if called from a subdirectory
        config_path = os.path.join("..", config_path)
        if not os.path.exists(config_path):
            config_path = os.path.join("..", "..", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_config_val(config, key_path, default=None):
    """Retrieves a value from a nested dictionary using a dot-separated path."""
    keys = key_path.split(".")
    val = config
    try:
        for k in keys:
            val = val[k]
        return val
    except (KeyError, TypeError):
        return default
