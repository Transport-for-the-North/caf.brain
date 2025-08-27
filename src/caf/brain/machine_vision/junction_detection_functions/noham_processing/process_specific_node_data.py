# -*- coding: utf-8 -*-
"""
Created on: 6/24/2025
Original author: Adil Zaheer
"""
# Built-Ins
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import os.path
from pathlib import Path

# Third Party
import pandas as pd


def process_noham(file_path: Path, output_path: Path):

    sheet_names = [
        "S Signal Zebra x Straight ahead",
        "S Classic Signalised (2+ arms)",
        "R",
        "PM Motorway (Straight on only)",
        "PM Motorway mergediverge",
        "ES (Signalised Rbout)",
        "EP (Giveway Rbout)",
        "P (Give way)",
        "M (Mini R)",
    ]

    for sheet in sheet_names:
        data = pd.read_excel(file_path, sheet_name=sheet)
        final_path = os.path.join(output_path, f"{sheet}.csv")
        data.to_csv(final_path)

    return


if __name__ == "__main__":
    process_noham(
        file_path=Path(
            r"E:\2025 work streams\caf.brAIn\machine vision\MVP work\input_image_processing\input\Junction types NoHAM 2018.xlsm"
        ),
        output_path=Path(
            r"E:\2025 work streams\caf.brAIn\machine vision\MVP work\input_image_processing\output\individual_junc_types"
        ),
    )
