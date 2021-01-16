#!/bin/bash
apt update && apt install -y libglib2.0-0 libgthread-2.0.so.0 libglib-2.0.so.0 libsm6 libxext6 libxrender-dev
pip install --upgrade pip
pip install opencv-python-headless
gunicorn --bind=0.0.0.0 --timeout 600 app:app
