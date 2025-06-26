# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/7/2025
Original author: Adil Zaheer
"""
import cv2
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
import os
import torch


def get_augmentation_pipeline():
    return A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.Affine(
             scale=(0.8, 1.2),  # zoom
             rotate=(-10, 10),  # rotation
             translate_percent=(-0.1, 0.1),
             p=0.5),
         # occlusion
         A.CoarseDropout(
             num_holes_range=(1, 8),
             hole_height_range=(0.02, 0.1),
             hole_width_range=(0.02, 0.1),
             fill=0,
             p=0.3
         ),
         # colour augmentations
         A.OneOf([
             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
             A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                  p=0.8)], p=0.7),
         # blur
         A.GaussianBlur(blur_limit=(3, 5), p=0.2),
         # noise
         A.GaussNoise(std_range=(0.01, 0.05),
                      mean_range=(0, 0),
                      per_channel=True,
                      p=0.2),
         # normalisation
         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                     max_pixel_value=255.0, p=1.0),
         # Convert to tensor for PyTorch
         ToTensorV2(p=1.0)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_labels(label_path):
    boxes = []
    class_labels = []

    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:5])

            boxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)

    return boxes, class_labels


def save_augmented_labels(output_label_path, boxes, class_labels):
    with open(output_label_path, 'w') as f:
        for box, class_id in zip(boxes, class_labels):
            x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def normalise_and_augment(image_dict, output):
    augmented_img_dir = os.path.join(output, 'augmented_images')
    augmented_label_dir = os.path.join(output, 'augmented_labels')

    os.makedirs(augmented_img_dir, exist_ok=True)
    os.makedirs(augmented_label_dir, exist_ok=True)

    transform = get_augmentation_pipeline()

    results = []
    images = []
    all_boxes = []
    all_class_labels = []
    image_paths = []

    for image_path, label_path in image_dict.items():
        base_name = os.path.basename(image_path)
        file_name = os.path.splitext(base_name)[0]

        output_tensor_path = os.path.join(augmented_img_dir, f"{file_name}.pt")
        output_label_path = os.path.join(augmented_label_dir, f"{file_name}_augmented.txt")

        if os.path.exists(output_tensor_path):
            data = torch.load(output_tensor_path)
            transformed_img = data['image']
            transformed_boxes = data['boxes']
            transformed_class_labels = data['labels']

        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes, class_labels = load_labels(label_path)

            transformed = transform(image=img, bboxes=boxes, class_labels=class_labels)
            transformed_img = transformed['image']
            transformed_boxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            save_augmented_labels(output_label_path, transformed_boxes, transformed_class_labels)

            if transformed_boxes:
                box_tensor = torch.tensor(transformed_boxes, dtype=torch.float32)
                label_tensor = torch.tensor(transformed_class_labels, dtype=torch.long)
            else:
                box_tensor = torch.zeros((0, 4), dtype=torch.float32)
                label_tensor = torch.zeros(0, dtype=torch.long)

            torch.save({
                'image': transformed_img,
                'boxes': box_tensor,
                'labels': label_tensor,
                'original_path': image_path
            }, output_tensor_path)

        for box, class_id in zip(transformed_boxes, transformed_class_labels):
            x_center, y_center, width, height = box
            results.append({
                'image_path': output_tensor_path,
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

        images.append(transformed_img)
        all_boxes.append(transformed_boxes)
        all_class_labels.append(transformed_class_labels)
        image_paths.append(image_path)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output, 'augmented_labels.csv'), index=False)

    return {'images': images,
            'boxes': all_boxes,
            'class_labels': all_class_labels,
            'image_paths': image_paths}
