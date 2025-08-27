# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/14/2025
Original author: Adil Zaheer
"""
# Built-Ins
import glob
import logging
import os

# Third Party
import pandas as pd
import yaml
from ultralytics import YOLO

LOG = logging.getLogger(__name__)


def prediction(output, final_model_path):
    LOG.info("Prediction running")
    yaml_file_path = os.path.join(output, "config.yaml")
    with open(yaml_file_path, "r") as f:
        config_data = yaml.safe_load(f)

    test_path = config_data.get("test", None)
    if not test_path:
        print("No test path found in config")
        return

    prediction_results_dir = os.path.join(output, "prediction_results_folder")
    os.makedirs(prediction_results_dir, exist_ok=True)

    model = YOLO(final_model_path)
    results = model.predict(
        source=test_path,
        save=True,  # annotated images
        save_txt=True,  # txt files with coordinates
        save_conf=True,  # confidence scores in txt files
        project=prediction_results_dir,
        name="predictions",
        show_labels=True,  # labels on images
        show_conf=True,  # confidence scores on images
        show_boxes=True,  # bounding boxes on images
        line_width=2,
    )
    LOG.info("Prediction finished")
    return prediction_results_dir, model


def extract_results(prediction_results_dir, model):
    LOG.info("Results extraction running")

    labels_dir = os.path.join(prediction_results_dir, "predictions", "labels")
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    data = []

    for txt_file in txt_files:
        image_name = os.path.basename(txt_file).replace(".txt", "")

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            # Format: [class_id] [x_center] [y_center] [width] [height] [confidence]
            class_id = int(float(parts[0]))
            confidence = float(parts[5]) if len(parts) > 5 else None

            class_name = model.names[class_id]

            data.append(
                {"image_name": image_name, "class": class_name, "confidence": confidence}
            )

    df = pd.DataFrame(data)

    csv_path = os.path.join(prediction_results_dir, "predictions", "prediction_summary.csv")
    df.to_csv(csv_path, index=False)
    LOG.info("Results extraction finished")

    return
