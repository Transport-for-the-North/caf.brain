"""
Created on: 10/10/2025
Original author: Adil Zaheer
"""
import os
from pathlib import Path


def model_building(output_path: Path,
                   training_images_path: Path,
                   classification_names: list[str],
                   hyperparameter_optimisation: str,
                    ):
    """

    Parameters
    ----------
    output_path
    training_images_path
    classification_names
    hyperparameter_optimisation

    Returns
    -------

    """
    main_output_folder = os.path.join(output_path, 'ModelBuildingOutputs')
    os.makedirs(main_output_folder, exist_ok=True)

    train_folder_name = os.path.join(main_output_folder, 'train')
    if not os.path.exists(train_folder_name):
        imageprocessor = BuildImagesTT(output=main_output_folder,
                                       image_dict=None,
                                       image_location=training_images_path)

        train, test, validation = imageprocessor.run_fullclass()

        imageprocessor.generate_folders(dictionary=train, folder_name='train')
        imageprocessor.generate_folders(dictionary=test, folder_name='test')
        imageprocessor.generate_folders(dictionary=validation, folder_name='val')

        train_path = os.path.join(main_output_folder, 'train')
        test_path = os.path.join(main_output_folder, 'test')
        val_path = os.path.join(main_output_folder, 'val')

        ensure_labels(folder_path=train_path)
        ensure_labels(folder_path=test_path)
        ensure_labels(folder_path=val_path)

    build_config(output=main_output_folder, class_names=classification_names)

    model_dir = os.path.join(main_output_folder, 'model_results')
    final_model_path = os.path.join(model_dir, 'best.pt')
    if not os.path.exists(final_model_path):
        main_model_build(output=main_output_folder,
                         hyperparameter_optimisation=hyperparameter_optimisation)
