"""
RoadSense: Utility functions and metadata constants for GTSRB.
"""

import os
import random
import numpy as np

# Official 43 GTSRB Class Names
CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles > 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for > 3.5t",
}

# Sign Categories for high-level grouped analysis
CLASS_CATEGORIES = {
    "Speed Limits": [0, 1, 2, 3, 4, 5, 7, 8],
    "Derestriction": [6, 32, 41, 42],
    "Prohibitory": [9, 10, 15, 16, 17],
    "Mandatory": [33, 34, 35, 36, 37, 38, 39, 40],
    "Danger / Warning": [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "Priority": [12, 13, 14],
}


def set_seed(seed: int = 42):
    """Fix random seeds across python, numpy, and framework for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_class_name(class_id: int) -> str:
    """Return readable sign name for a given class ID."""
    return CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
