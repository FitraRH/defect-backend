# models/model_loader.py
"""
Model loader with automatic model detection and loading
"""

import os
import torch
from anomalib.deploy import TorchInferencer
from .hrnet_model import create_hrnet_model
from config import *


class ModelLoader:
    """Handles automatic model loading and initialization"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.anomalib_model = None
        self.hrnet_model = None
        self.models_loaded = False
        
    def load_models(self, anomalib_path=None, hrnet_path=None):
        """
        Load models automatically or from specified paths
        
        Args:
            anomalib_path: Custom path to Anomalib model (optional)
            hrnet_path: Custom path to HRNet model (optional)
        """
        print("Loading detection models...")
        
        # Use custom paths if provided, otherwise use config defaults
        anomalib_model_path = anomalib_path or ANOMALIB_MODEL_PATH
        hrnet_model_path = hrnet_path or HRNET_MODEL_PATH
        
        # Load Anomalib model
        anomalib_success = self._load_anomalib_model(anomalib_model_path)
        
        # Load HRNet model
        hrnet_success = self._load_hrnet_model(hrnet_model_path)
        
        self.models_loaded = anomalib_success and hrnet_success
        
        if self.models_loaded:
            print("All models loaded successfully!")
        else:
            print("Some models failed to load. Check model paths.")
            
        return self.models_loaded
    
    def _load_anomalib_model(self, model_path):
        """Load Anomalib model"""
        if not os.path.exists(model_path):
            print(f"Anomalib model not found: {model_path}")
            return False
            
        try:
            print(f"Loading Anomalib model from {model_path}...")
            self.anomalib_model = TorchInferencer(path=model_path, device=self.device)
            print(f"Anomalib model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading Anomalib model: {e}")
            return False
    
    def _load_hrnet_model(self, model_path):
        """Load HRNet model"""
        if not os.path.exists(model_path):
            print(f"HRNet model not found: {model_path}")
            return False
            
        try:
            print(f"Loading HRNet model from {model_path}...")
            
            # Create model
            self.hrnet_model = create_hrnet_model(num_classes=6)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights
            try:
                self.hrnet_model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                print("Strict loading failed, trying flexible loading...")
                self.hrnet_model.load_state_dict(state_dict, strict=False)
            
            self.hrnet_model.to(self.device)
            self.hrnet_model.eval()
            
            print(f"HRNet model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading HRNet model: {e}")
            return False
    
    def get_models(self):
        """Return loaded models"""
        if not self.models_loaded:
            print("Models not loaded yet. Call load_models() first.")
            return None, None
        return self.anomalib_model, self.hrnet_model
    
    def is_ready(self):
        """Check if models are ready for inference"""
        return self.models_loaded and self.anomalib_model is not None and self.hrnet_model is not None


def auto_load_models(device='cuda'):
    """
    Convenience function to automatically load models
    
    Returns:
        tuple: (anomalib_model, hrnet_model) or (None, None) if failed
    """
    loader = ModelLoader(device=device)
    
    if loader.load_models():
        return loader.get_models()
    else:
        return None, None


def load_custom_models(anomalib_path, hrnet_path, device='cuda'):
    """
    Load models from custom paths
    
    Args:
        anomalib_path: Path to Anomalib model
        hrnet_path: Path to HRNet model
        device: Device to load models on
        
    Returns:
        tuple: (anomalib_model, hrnet_model) or (None, None) if failed
    """
    loader = ModelLoader(device=device)
    
    if loader.load_models(anomalib_path, hrnet_path):
        return loader.get_models()
    else:
        return None, None