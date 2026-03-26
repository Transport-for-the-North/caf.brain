"""
caf.brAIn is Transport for the North's bespoke machine learning and AI
library. It consists of:
- A generalised end to end machine learning pipeline
- Simplified machine learning functions
- Generalised end to end object detection machine vision model
- Simplified object detection functions
"""

from caf.brain.object_detection._object_detection import (
    generate_satellite_images,
    train_object_detection_model,
)
