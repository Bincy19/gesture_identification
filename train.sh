#!/bin/bash
# Quick start: train on up to 500 images per class, then run the webcam demo.
# Remove --max_images_per_class to use the full dataset (takes hours on CPU).
set -e
python3 retrain.py --image_dir dataset --max_images_per_class 500
python3 app.py
