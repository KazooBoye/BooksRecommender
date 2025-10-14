"""
Configuration management utilities.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value with dot notation support."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration structure."""
        required_sections = ['model', 'data', 'training', 'inference']
        
        for section in required_sections:
            if section not in self.config:
                print(f"Missing required section: {section}")
                return False
        
        # Validate model config
        model_config = self.config['model']
        required_model_keys = ['embedding_dim', 'num_layers', 'num_heads']
        for key in required_model_keys:
            if key not in model_config:
                print(f"Missing required model config: {key}")
                return False
        
        # Validate data config
        data_config = self.config['data']
        required_data_keys = ['processed_path', 'max_sequence_length']
        for key in required_data_keys:
            if key not in data_config:
                print(f"Missing required data config: {key}")
                return False
        
        return True


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories based on configuration."""
    directories = [
        config['data']['processed_path'],
        config['logging']['log_dir'],
        'models',
        'results',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    import logging
    from datetime import datetime
    
    log_dir = config['logging']['log_dir']
    log_level = config['logging']['level']
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"Logging configured. Log file: {log_file}")


def get_device_config() -> str:
    """Get optimal device configuration."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Apple Metal Performance Shaders (MPS) available")
        else:
            device = "cpu"
            print("Using CPU")
        
        return device
    
    except ImportError:
        print("PyTorch not available, using CPU")
        return "cpu"


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'scikit-learn',
        'yaml',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("All required dependencies are available")
    return True


def create_experiment_config(base_config_path: str, experiment_name: str, 
                           overrides: Dict[str, Any]) -> str:
    """Create experiment-specific configuration."""
    config_manager = ConfigManager(base_config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        config_manager.update(key, value)
    
    # Save experiment config
    experiment_dir = os.path.join('experiments', experiment_name)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    
    experiment_config_path = os.path.join(experiment_dir, 'config.yaml')
    config_manager.save(experiment_config_path)
    
    # Save experiment metadata
    metadata = {
        'experiment_name': experiment_name,
        'base_config': base_config_path,
        'overrides': overrides,
        'created_at': str(datetime.now())
    }
    
    metadata_path = os.path.join(experiment_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Experiment config created: {experiment_config_path}")
    return experiment_config_path


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager('configs/model_config.yaml')
    
    print("Configuration validation:", config_manager.validate())
    print("Model embedding dimension:", config_manager.get('model.embedding_dim'))
    print("Device configuration:", get_device_config())
    print("Dependencies check:", check_dependencies())