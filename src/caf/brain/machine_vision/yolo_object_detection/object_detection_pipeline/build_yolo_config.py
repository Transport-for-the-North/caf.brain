# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/8/2025
Original author: Adil Zaheer
"""
import yaml
from pathlib import Path
import os

def build_config(output: Path, class_names: list):
    train_dir = os.path.join(output, 'train')
    test_dir = os.path.join(output, 'test')
    val_dir = os.path.join(output, 'val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError("Train and Test folder should already be \
                                 be generated using the BuildImagesTT class")

    yaml_content = {
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'nc': int(len(class_names)),
        'names': class_names
    }

    yaml_file_path = os.path.join(output, 'config.yaml')

    with open(yaml_file_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False, default_flow_style=False)

    return
