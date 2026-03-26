"""
Created on: 11/10/2025
Original author: Adil Zaheer
"""

# Built-Ins
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

# Third Party
import pandas as pd
import yaml

LOG = logging.getLogger(__name__)


def _create_dict(user_images_folder_path: Path, output_path: Path) -> dict:
    """
    Create a lookup dictionary of labeled model building images.

    The key is the image path and the value is the bounding box path.

    Parameters
    ----------
    user_images_folder_path:
        Path to folder of user images. Labeling text files must exist at
        file location.
    output_path:
        Path to output folder.

    Returns
    -------
    Dictionary of model building images.
    """
    dictionary = {}
    img_dir = Path(user_images_folder_path)
    image_paths = list(img_dir.rglob("*.jpg"))
    no_label_images = []

    for image_path in image_paths:
        txt_path = Path(image_path).with_suffix(".txt")
        if txt_path.exists():
            dictionary[image_path] = txt_path

            LOG.info("Found image-label pair: %s -> %s", image_path, txt_path)
        else:
            LOG.warning("No image label pair found for %s", image_path)
            no_label_images.append(image_path)

    if no_label_images:
        problem_path = output_path / "no_label_pair.csv"
        no_label_df = pd.DataFrame({"image_path": no_label_images, "status": "missing_label"})
        no_label_df.to_csv(problem_path, index=False)
        LOG.info("No label image pairs found and exported to %s", problem_path)

    if len(no_label_images) > 0.25 * len(image_paths):
        raise ValueError(
            "Over 25% of your images do not contain any labels. \n"
            "Please review these images (paths in no_label_pair.csv). If these images are of a suitable \n"
            "quality then label them and re run the model. If they are problematic images then remove \n"
            "these from the folder and re run the function."
        )

    return dictionary


def _split_dict(
    dictionary: dict, split_ratio: float = 0.15
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """
    Split a dictionary into training and validation sets.

    Parameters
    ----------
    dictionary:
        Generated from _create_dict(). This dictionary should have the key as
        the image path and the value as the bounding box text file path.
    split_ratio: Percentage of images that go into the other set e.g 0.15
                 would be 85% in train, 15% in test.

    Returns
    -------
    Two dictionaries comprised of the data in the input dictionary split by
    the provided ratio.
    """
    split_index = int(len(dictionary) * split_ratio)
    selected_keys = random.sample(list(dictionary.keys()), split_index)
    test = {key: dictionary[key] for key in selected_keys}
    train = {key: dictionary[key] for key in dictionary if key not in selected_keys}

    return train, test


def generate_folders(dictionary: dict, folder_name: str, output: Path) -> None:
    """
    Copy image/bbox pairs into a subfolder under the output directory.

    Parameters
    ----------
    dictionary:
        Mapping of image paths (keys) to bounding box text paths (values).
    folder_name:
        Name of the subfolder to create inside output.
    output:
        Root directory where the subfolder and copied files are placed.

    Returns
    -------
    None
    """
    counter = 0
    for key, value in dictionary.items():
        new_img_dir = output / folder_name
        new_img_dir.mkdir(parents=True, exist_ok=True)

        base_name = os.path.basename(key)
        file_name = os.path.splitext(base_name)[0]

        img_output_path = new_img_dir / f"{file_name}.jpg"
        bbox_output_path = new_img_dir / f"{file_name}.txt"

        shutil.copy(key, img_output_path)
        shutil.copy(value, bbox_output_path)

        counter += 1
        if counter % 10 == 0:
            LOG.info("Processed %d/%d items for %s", counter, len(dictionary), folder_name)


def ensure_labels(folder_path: Path) -> None:
    """
    Remove images without valid labels.

    Scans a folder for .txt label files. If a label file is empty,
    the function deletes both the label file and its corresponding
    .jpg image (same base name).

    Parameters
    ----------
    folder_path:
        Path to the dataset folder to check.

    Returns
    -------
    None
    """
    txt_paths = list(Path(folder_path).rglob("*.txt"))
    for txt_file in txt_paths:
        if os.path.getsize(txt_file) == 0:
            base_name = os.path.splitext(txt_file)[0]
            jpg_path = base_name + ".jpg"

            if os.path.exists(jpg_path):
                os.remove(txt_file)
                os.remove(jpg_path)
                print(f"Removed: {txt_file} and {jpg_path}")
            else:
                print(f"Warning: {jpg_path} not found for {txt_file}")


def build_config(output: Path, class_names: list) -> None:
    """
    Builds YAML config file required for YOLO object detection.

    Parameters
    ----------
    output:
        Path to the output folder.
    class_names:
        Classification names in the order they were labelled in.

    Returns
    -------
    None
    """
    train_dir = output / "train"
    test_dir = output / "test"
    val_dir = output / "val"

    if not train_dir.exists():
        raise FileNotFoundError("Train and Test folders should already exist")

    yaml_content = {
        "train": str(train_dir),
        "val": str(val_dir),
        "test": str(test_dir),
        "nc": int(len(class_names)),
        "names": class_names,
    }

    yaml_file_path = output / "config.yaml"
    with open(yaml_file_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml_content, file, sort_keys=False, default_flow_style=False)


def count_valid_pairs(folder_path: Path) -> dict:
    """
    Function to ensure enough training images were created.

    Parameters
    ----------
    folder_path:
        Path to user images for model training.

    Returns
    -------
    Dict of image amount information.
    """
    folder = Path(folder_path)
    images = {f.stem for f in folder.glob("*.png")}
    labels = {f.stem for f in folder.glob("*.txt")}

    valid = images & labels
    missing_labels = images - labels
    missing_images = labels - images

    return {
        "total_images": len(images),
        "valid_pairs": len(valid),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
    }


def main_ttv_creation(
    user_images_folder_path: Path, output_path: Path
) -> tuple[dict, dict, dict]:
    """
    Creates train, test and validation data.

    Based on a folder path of images. These images should have been
    generated by generate_satellite_images.

    Parameters
    ----------
    user_images_folder_path:
        Path to folder of user images.
    output_path:
        Path to output folder.

    Returns
    -------
    Train, test and validation dictionaries.
    """

    image_dict = _create_dict(
        user_images_folder_path=user_images_folder_path, output_path=output_path
    )
    train, test = _split_dict(dictionary=image_dict)
    final_train, validation = _split_dict(dictionary=train)

    return final_train, test, validation
