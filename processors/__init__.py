# processors/__init__.py
"""
Processors package for Unified Defect Detection System
"""

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor

__all__ = ['ImageProcessor', 'VideoProcessor']