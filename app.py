from flask import Flask, request
from flask import Blueprint, jsonify
import numpy as np
import tensorflow as tf
import onnxruntime
import cv2
import boto3
from botocore.exceptions import NoCredentialsError
from thresholdingfunction import otsuthresholding
from classify import load_model, load_image_fromnumpy, predict_single
from torchvision import datasets, models, transforms
import torch
import os
from blackandwhiteratios import blackandwhiteratio
from boundingbox import cropImage

from helpers import (load_image, make_square, 
                     augment, pre_process, softmax)
from helper_config import (IMG_HEIGHT, IMG_WIDTH, CLASS_MAP,
                           CHANNELS)

from datetime import datetime


##ACCESS_KEY ='AKIA2U5YERTOYQY77S5M'

##SECRET_KEY = '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'
##BUCKET='photostakenduringpilotstudy'

# Usually helps in debugging
print(tf.__version__) # Print the version of tensorflow being used

app = Flask(__name__)

moz = Blueprint('moz', __name__)

prep = pre_process(IMG_WIDTH, IMG_HEIGHT)

@moz.route("/get_label", methods=['GET', 'POST'])
def get_label():
    inf_file = request.files.get('image')
    ##inf_file = request.files.get('image').read()
    MosqID = request.form.get('MosquitoID')
    PicNum = request.form.get('PictureNumber')
    SiteID = request.form.get('SiteID')
    inf_fileread = request.files.get('image').read()
    print("Got the file")
    label = run_inference(inf_fileread)
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    fname = date_time + "_" + MosqID + "_" + PicNum + "_" + SiteID + "_" + label[0] + "_" + label[1] + "_attatchedlens.jpg"
    ##fname = "mypic.jpg"
    #s3 = boto3.client('s3',aws_access_key_id= ACCESS_KEY, aws_secret_access_key= SECRET_KEY)
    #status=upload_to_aws(inf_file, BUCKET, fname)
    #print(status)
    
    return jsonify({
        "genus": label[0],
        "species": label[1],
        "confidence_score": label[2],
        "color_code": label[3]
    })
  
def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_fileobj(local_file, bucket, s3_file)
        ##s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def color_code(num):
    if(float(num)>0.9):
        return '#4bf542'
    elif (float(num)>0.7):
        return '#f7b17e'
    else:
        return '#f7543b'

def run_inference(inf_file):
    # Preprocessing of the image happens here
    useless, img, status=cropImage(impath, 'm', labelsfile, 21, 0.08)
    print("Image Loaded")
    img = make_square(img)
    img = augment(prep, img)
    print("Transformations done")
    img = img.transpose(-1, 0, 1).astype(np.float32)
    img = img.reshape(-1, CHANNELS, IMG_WIDTH, IMG_HEIGHT)

    # Inferencing starts here
    sess = onnxruntime.InferenceSession("./best_acc.onnx")
    print("The model expects input shape: ", sess.get_inputs()[0].shape)
    print("The shape of the Image is: ", img.shape)
    input_name = sess.get_inputs()[0].name

    result = sess.run(None, {input_name: img})
    prob_array = result[0][0]
    print("Prob Array ", prob_array)
    prob = max(softmax(result[0][0]))
    print("Prob ",prob)
    species = tf.argmax(prob_array.ravel()[:20]).numpy()
    print("Class Label ", species)
    print("Spec ", CLASS_MAP[species][1])
    string_label = CLASS_MAP[species][1].split(" ")
    return (string_label[0], string_label[1], str(prob), color_code(prob))

app.register_blueprint(moz)
