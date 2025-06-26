# -*- coding: utf-8 -*-
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
"""
Created on: 5/6/2025
Original author: Adil Zaheer
"""
import os
from pathlib import Path
import glob
import shutil
import random
import logging
import joblib
import pandas as pd
LOG = logging.getLogger(__name__)


class BuildImagesTT:
    """
Need to fill out
    """
    def __init__(self, output: Path, image_location: Path, image_dict: dict, split_ratio=0.15):
        self.output = output
        self.split_ratio = split_ratio
        self.image_dict = image_dict
        self.image_location = image_location

    def run_fullclass(self):
        """
        Runs full class
        Returns
        -------

        """
        image_dict = self.create_dict()
        train, test = self.split_dict(dictionary=image_dict)
        final_train, validation = self.split_dict(dictionary=train)

        if self.output is not None:
            train_dict_name = os.path.join(self.output, 'train_dict.pkl')
            joblib.dump(final_train, train_dict_name)

            val_dict_name = os.path.join(self.output, 'val_dict.pkl')
            joblib.dump(validation, val_dict_name)

            test_dict_name = os.path.join(self.output, 'test_dict.pkl')
            joblib.dump(test, test_dict_name)

        return final_train, test, validation


    def create_label_data(self, image_dict):
        """

        Parameters
        ----------
        image_dict

        Returns
        -------

        """
        label_list = []
        for _, value in image_dict.items():
            with open(os.path.join(value), 'r', encoding='UTF-8') as file:
                lines = file.readlines()
                for line in lines:
                    label_list.append(line[0])

        counts = pd.Series(label_list).value_counts()
        final = counts.to_dict()
        df = pd.DataFrame.from_dict(final, orient='index', columns=['occurrences'])
        df = df.reset_index().rename(columns={'index': 'labels'})
        df.to_csv(os.path.join(self.output, 'label_occurrence_info.csv'), index=False)

        return df


    def create_dict(self):
        """
        
        Returns
        -------

        """
        # key: image path
        # value: bounding box path
        dictionary = {}
        img_dir = os.path.join(self.image_location)
        image_paths = glob.glob(os.path.join(img_dir) + '/**/*.jpg', recursive=True)

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            txt_name = base_name.replace('.jpg', '.txt')

            txt_path = os.path.join(os.path.dirname(image_path), txt_name)
            if os.path.exists(txt_path):
                dictionary[image_path] = txt_path
                print(f"Found image-label pair: {image_path} -> {txt_path}")

        return dictionary


    def split_dict(self, dictionary):
        """

        Parameters
        ----------
        dictionary

        Returns
        -------

        """
        # how many items go into test with new method
        split_index = int(len(dictionary) * self.split_ratio)
        # items = list(dictionary.items())
        # train = dict(items[:split_index])
        # test = dict(items[split_index:])

        selected_keys = random.sample(list(dictionary.keys()), split_index)

        test = {key: dictionary[key] for key in selected_keys}
        train = {key: dictionary[key] for key in dictionary if key not in selected_keys}

        return train, test


    def generate_folders(self, dictionary, folder_name):
        """

        Parameters
        ----------
        dictionary
        folder_name

        Returns
        -------

        """
        counter = 0
        for key, value in dictionary.items():
            new_img_dir = os.path.join(self.output, f'{folder_name}')
            os.makedirs(new_img_dir, exist_ok=True)

            base_name = os.path.basename(key)
            file_name = os.path.splitext(base_name)[0]

            img_output_path = os.path.join(new_img_dir, f"{file_name}.jpg")
            bbox_output_path = os.path.join(new_img_dir, f"{file_name}.txt")

            # os.replace(key, img_output_path)
            # os.replace(value, bbox_output_path)
            shutil.copy(key, img_output_path)
            shutil.copy(value, bbox_output_path)

            counter += 1
            if counter % 10 == 0:
                LOG.info(f"Processed {counter}/{len(dictionary)} items for {folder_name}")


def ensure_labels(folder_path):
    """
    check that all the images actually have labels
    go into text files, if empty. remove the image and the text file from the folder (same names)

    Parameters
    ----------
    folder_path

    Returns
    -------

    """
    txt_paths = glob.glob(os.path.join(folder_path, '**/*.txt'), recursive=True)
    for txt_file in txt_paths:
        if os.path.getsize(txt_file) == 0:
            base_name = os.path.splitext(txt_file)[0]
            jpg_path = base_name + '.jpg'

            if os.path.exists(jpg_path):
                os.remove(txt_file)
                os.remove(jpg_path)
                print(f"Removed: {txt_file} and {jpg_path}")
            else:
                print(f"Warning: {jpg_path} not found for {txt_file}")
