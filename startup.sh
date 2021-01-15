#!/bin/bash
apt update && apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
pip install --upgrade pip
pip uninstall opencv-python
pip install opencv-python
gunicorn --bind=0.0.0.0 --timeout 600 app:app
