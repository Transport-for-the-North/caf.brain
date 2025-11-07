"""
Created on: 03/11/2025
Original author: Adil Zaheer
"""

import logging
import time
import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import torch

LOG = logging.getLogger(__name__)


def _prediction(
    output: Path, final_model_path: Path, prediction_images: Path
) -> tuple[Path, YOLO]:
    """
    Run predictions on a folder of images using a trained YOLO model.

    Parameters
    ----------
    output: Root directory where prediction results will be stored.
    final_model_path: Path to the trained YOLO model weights (e.g., best.pt).
    prediction_images: Path to a folder of images to run predictions on.

    Returns
    -------
    prediction_results_dir: Path to the directory containing prediction outputs.
    model: The YOLO model instance used for prediction.
    """
    start_time = time.time()
    LOG.info("Prediction running")

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        LOG.info("GPU available and being used to run the model")
    else:
        LOG.warning("GPU not available. CPU being used.")
        torch.set_num_threads(8)

    output = Path(output)
    final_model_path = Path(final_model_path)
    prediction_images = Path(prediction_images)

    prediction_results_dir = output / "prediction_results_folder"
    prediction_results_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(final_model_path)
    _ = model.predict(
        source=prediction_images,
        save=True,  # annotated images
        save_txt=True,  # txt files with coordinates
        save_conf=True,  # confidence scores in txt files
        project=prediction_results_dir,
        name="predictions",
        show_labels=True,  # labels on images
        show_conf=True,  # confidence scores on images
        show_boxes=True,  # bounding boxes on images
        line_width=2,
        device=device,
    )

    end_time = time.time()
    LOG.info("Total prediction run time: %.2f seconds", end_time - start_time)
    LOG.info("Prediction finished")
    return prediction_results_dir, model


def _extract_results(prediction_results_dir: Path, model: YOLO) -> None:
    """
    Reads YOLO prediction label files from the results directory, parses
    class IDs and confidence scores, maps them to class names, and writes
    a summary CSV file (prediction_summary.csv).

    Parameters
    ----------
    prediction_results_dir: Directory containing YOLO prediction outputs
                            (with predictions/labels).
    model: The YOLO model instance, used to map class IDs to class names.

    Returns
    -------
    None
    """
    LOG.info("Results extraction running")

    prediction_results_dir = Path(prediction_results_dir)
    labels_dir = prediction_results_dir / "predictions" / "labels"
    txt_files = list(labels_dir.glob("*.txt"))

    data = []

    for txt_file in txt_files:
        image_name = os.path.basename(txt_file).replace(".txt", "")

        with open(txt_file, "r", encoding="utf-8") as f:
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

    csv_path = prediction_results_dir / "predictions" / "prediction_summary.csv"
    df.to_csv(csv_path, index=False)
    LOG.info("Results extraction finished")


def main_prediction(output: Path, images_to_predict_path: Path, model_path: Path) -> None:
    """
    Run the full prediction pipeline for a trained YOLO model.

    Steps:
    1. Validate inputs (model path, output directory, images folder, class names).
    2. Build a prediction configuration file (`config.yaml`) with class names.
    3. Run YOLO predictions on the provided images using the trained model.
    4. Extract results into a summary CSV file.

    Parameters
    ----------
    output: Root directory where prediction results will be stored.
    images_to_predict_path: Path to the file containing prediction coordinates.
                            This can be either a geometry column as part of a
                            shapefile or a csv with an x and y column.
    model_path: Path to the trained YOLO model weights (e.g., best.pt).

    Returns
    -------
    None
    """

    if model_path is None:
        raise ValueError(
            "Model_path is none. This is required in order to run \
             the model. If you ran the model build it should be located \
             inside a folder your_out_path/ModelBuildingOutputs/model_results/best.pt. \
             This is generated when the model is trained which is what \
             you are using for your predictions. If you did not \
             run the model build then it should be located here: \
             "
        )

    if output is None:
        raise ValueError(
            "Please provide an output path. This should be a \
                          location on your computers drive e.g. E:/Documents"
        )

    if images_to_predict_path is None:
        raise ValueError(
            "Please ensure prediction_object_detection has been \
                          correctly run as this should create the images to \
                          predict."
        )

    prediction_results_dir, model = _prediction(
        output=output, final_model_path=model_path, prediction_images=images_to_predict_path
    )

    _extract_results(prediction_results_dir=prediction_results_dir, model=model)
