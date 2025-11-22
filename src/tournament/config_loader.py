"""Configuration loader for tournament settings."""
import json
from pathlib import Path
from typing import Optional, Dict, Any


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load tournament configuration from a JSON file.
    
    Args:
        config_path: Path to the config file. If None, looks for tournament_config.json
                    in the project root directory.
    
    Returns:
        Dictionary containing configuration values with defaults applied.
    """
    if config_path is None:
        # Look for tournament_config.json in the project root
        # This file is in src/tournament/, so we go up 2 levels
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "tournament_config.json"
    else:
        config_path = Path(config_path)
    
    # Default values
    defaults = {
        "num_bots": 500,
        "games_per_bot": 100,
        "num_generations": 1,
        "num_workers": None,
        "survivors_per_generation": 250,
        "mutation_chance": 0.1,
        "mutation_amount": 0.5,
    }
    
    # If config file doesn't exist, return defaults
    if not config_path.exists():
        return defaults
    
    # Load config file
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        # Merge with defaults (config file values override defaults)
        result = defaults.copy()
        for key, value in config_data.items():
            if key in defaults:
                # Convert null to None
                if value is None:
                    result[key] = None
                else:
                    result[key] = value
        
        return result
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        print("Using default values.")
        return defaults

