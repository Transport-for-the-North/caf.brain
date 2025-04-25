# -*- coding: utf-8 -*-
"""
Created on: 4/4/2025
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position

from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data='path/to/data.yaml', epochs=50, imgsz=640, weights='yolo11n.pt')
results = model("path/to/test_satellite_image.jpg")
results.show()




