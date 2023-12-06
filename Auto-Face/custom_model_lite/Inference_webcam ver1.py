######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

import datetime
now = datetime.datetime.now()
date_time = now.strftime("%Y-%m-%d %H:%M:%S")
import xml.etree.ElementTree as ET

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
face_detected_count = 0
captured_Lord_John_Perucho = False
captured_Leo_Delen = False
captured_Frank_Lester_Castillo = False
captured_Reu_Pan = False
captured_Queenie_Rose_Amargo = False

target_reset_time = datetime.time(1, 16)

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

lord_john_perucho_counter = 0

LJ_Folder = 'Face-Detected-LJ'  
if not os.path.exists(LJ_Folder):
    os.makedirs(LJ_Folder)

Leo_Folder = 'Face-Detected-Leo'  
if not os.path.exists(Leo_Folder):
    os.makedirs(Leo_Folder)

Kiko_Folder = 'Face-Detected-Kiko'  
if not os.path.exists(Kiko_Folder):
    os.makedirs(Kiko_Folder)

Queenie_Folder = 'Face-Detected-Queenie'  
if not os.path.exists(Queenie_Folder):
    os.makedirs(Queenie_Folder)

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        current_datetime = datetime.datetime.now()
        current_time1 = current_datetime.time()
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(0.5,(boxes[i][0] * imH)))
            xmin = int(max(0.5,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            object_name = labels[int(classes[i])] 
            print("Object Name:", object_name)
            object_name = labels[int(classes[i])]

            if object_name == "Lord John Perucho":
                face_detected_count += 1

                if face_detected_count == 20:
                    object_name = labels[int(classes[i])] 
                    label = '%s: DateTime: %s' % (object_name, date_time) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                    label_ymin = max(ymin, labelSize[1] + 10) 
                    current_time = time.strftime("%Y-%m-%d %H-%M-%S")
                    #For Saving Annotation
                    image_name = f"{current_time} {object_name}.jpg"
                    image_path = os.path.join(LJ_Folder, image_name)
                    cv2.imwrite(image_path, frame) 
                    print("Image captured and saved!")
                    face_detected_count = 0
                    captured_Lord_John_Perucho = True
                    #For Annotation
                    annotation = ET.Element('annotation')
                    folder = ET.SubElement(annotation, 'folder')
                    folder.text = 'Face-Detected'
                    filename = ET.SubElement(annotation, 'filename')
                    filename.text = image_name
                    path = ET.SubElement(annotation, 'path')
                    path.text = os.path.abspath(image_path)
                    source = ET.SubElement(annotation, 'source')
                    database = ET.SubElement(source, 'database')
                    database.text = 'Unknown'
                    size = ET.SubElement(annotation, 'size')
                    width_elem = ET.SubElement(size, 'width')
                    width_elem.text = str(320)
                    height_elem = ET.SubElement(size, 'height')
                    height_elem.text = str(320)
                    depth = ET.SubElement(size, 'depth')
                    depth.text = str(3)
                    segmented = ET.SubElement(annotation, 'segmented')
                    segmented.text = '0'
                    obj = ET.SubElement(annotation, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'Lord John Perucho'
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '1'
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin_elem = ET.SubElement(bndbox, 'xmin')
                    xmin_elem.text = str(xmin)
                    ymin_elem = ET.SubElement(bndbox, 'ymin')
                    ymin_elem.text = str(ymin)
                    xmax_elem = ET.SubElement(bndbox, 'xmax')
                    xmax_elem.text = str(xmax)
                    ymax_elem = ET.SubElement(bndbox, 'ymax')
                    ymax_elem.text = str(ymax)
                    #For Saving Annotation
                    xml_filename = f"{current_time} {object_name}.xml"
                    xml_path = os.path.join(LJ_Folder, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    print("Annotation XML file saved!")

            if object_name == "Leo Delen":
                face_detected_count += 1

                if face_detected_count == 20:
                    object_name = labels[int(classes[i])] 
                    label = '%s: DateTime: %s' % (object_name, date_time) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                    label_ymin = max(ymin, labelSize[1] + 10) 
                    current_time = time.strftime("%Y-%m-%d %H-%M-%S")
                    #For Saving Annotation
                    image_name = f"{current_time} {object_name}.jpg"
                    image_path = os.path.join(Leo_Folder, image_name)
                    cv2.imwrite(image_path, frame) 
                    print("Image captured and saved!")
                    face_detected_count = 0
                    captured_Leo_Delen = True
                    #For Annotation
                    annotation = ET.Element('annotation')
                    folder = ET.SubElement(annotation, 'folder')
                    folder.text = 'Face-Detected'
                    filename = ET.SubElement(annotation, 'filename')
                    filename.text = image_name
                    path = ET.SubElement(annotation, 'path')
                    path.text = os.path.abspath(image_path)
                    source = ET.SubElement(annotation, 'source')
                    database = ET.SubElement(source, 'database')
                    database.text = 'Unknown'
                    size = ET.SubElement(annotation, 'size')
                    width_elem = ET.SubElement(size, 'width')
                    width_elem.text = str(320)
                    height_elem = ET.SubElement(size, 'height')
                    height_elem.text = str(320)
                    depth = ET.SubElement(size, 'depth')
                    depth.text = str(3)
                    segmented = ET.SubElement(annotation, 'segmented')
                    segmented.text = '0'
                    obj = ET.SubElement(annotation, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'Leo Delen'
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '1'
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin_elem = ET.SubElement(bndbox, 'xmin')
                    xmin_elem.text = str(xmin)
                    ymin_elem = ET.SubElement(bndbox, 'ymin')
                    ymin_elem.text = str(ymin)
                    xmax_elem = ET.SubElement(bndbox, 'xmax')
                    xmax_elem.text = str(xmax)
                    ymax_elem = ET.SubElement(bndbox, 'ymax')
                    ymax_elem.text = str(ymax)
                    #For Saving Annotation
                    xml_filename = f"{current_time} {object_name}.xml"
                    xml_path = os.path.join(Leo_Folder, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    print("Annotation XML file saved!")

            if object_name == "Frank Lester castillo":
                face_detected_count += 1

                if face_detected_count == 20:
                    object_name = labels[int(classes[i])] 
                    label = '%s: DateTime: %s' % (object_name, date_time) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                    label_ymin = max(ymin, labelSize[1] + 10) 
                    current_time = time.strftime("%Y-%m-%d %H-%M-%S")
                    #For Saving Annotation
                    image_name = f"{current_time} {object_name}.jpg"
                    image_path = os.path.join(Kiko_Folder, image_name)
                    cv2.imwrite(image_path, frame) 
                    print("Image captured and saved!")
                    face_detected_count = 0
                    captured_Lord_John_Perucho = True
                    #For Annotation
                    annotation = ET.Element('annotation')
                    folder = ET.SubElement(annotation, 'folder')
                    folder.text = 'Face-Detected'
                    filename = ET.SubElement(annotation, 'filename')
                    filename.text = image_name
                    path = ET.SubElement(annotation, 'path')
                    path.text = os.path.abspath(image_path)
                    source = ET.SubElement(annotation, 'source')
                    database = ET.SubElement(source, 'database')
                    database.text = 'Unknown'
                    size = ET.SubElement(annotation, 'size')
                    width_elem = ET.SubElement(size, 'width')
                    width_elem.text = str(320)
                    height_elem = ET.SubElement(size, 'height')
                    height_elem.text = str(320)
                    depth = ET.SubElement(size, 'depth')
                    depth.text = str(3)
                    segmented = ET.SubElement(annotation, 'segmented')
                    segmented.text = '0'
                    obj = ET.SubElement(annotation, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'Frank Lester castillo'
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '1'
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin_elem = ET.SubElement(bndbox, 'xmin')
                    xmin_elem.text = str(xmin)
                    ymin_elem = ET.SubElement(bndbox, 'ymin')
                    ymin_elem.text = str(ymin)
                    xmax_elem = ET.SubElement(bndbox, 'xmax')
                    xmax_elem.text = str(xmax)
                    ymax_elem = ET.SubElement(bndbox, 'ymax')
                    ymax_elem.text = str(ymax)
                    #For Saving Annotation
                    xml_filename = f"{current_time} {object_name}.xml"
                    xml_path = os.path.join(Kiko_Folder, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    print("Annotation XML file saved!")

            if object_name == "Queenie Rose Amargo ":
                face_detected_count += 1

                if face_detected_count == 20:
                    object_name = labels[int(classes[i])] 
                    label = '%s: DateTime: %s' % (object_name, date_time) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                    label_ymin = max(ymin, labelSize[1] + 10) 
                    current_time = time.strftime("%Y-%m-%d %H-%M-%S")
                    #For Saving Annotation
                    image_name = f"{current_time} {object_name}.jpg"
                    image_path = os.path.join(Queenie_Folder, image_name)
                    cv2.imwrite(image_path, frame) 
                    print("Image captured and saved!")
                    face_detected_count = 0
                    captured_Queenie_Rose_Amargo = True
                    #For Annotation
                    annotation = ET.Element('annotation')
                    folder = ET.SubElement(annotation, 'folder')
                    folder.text = 'Face-Detected'
                    filename = ET.SubElement(annotation, 'filename')
                    filename.text = image_name
                    path = ET.SubElement(annotation, 'path')
                    path.text = os.path.abspath(image_path)
                    source = ET.SubElement(annotation, 'source')
                    database = ET.SubElement(source, 'database')
                    database.text = 'Unknown'
                    size = ET.SubElement(annotation, 'size')
                    width_elem = ET.SubElement(size, 'width')
                    width_elem.text = str(320)
                    height_elem = ET.SubElement(size, 'height')
                    height_elem.text = str(320)
                    depth = ET.SubElement(size, 'depth')
                    depth.text = str(3)
                    segmented = ET.SubElement(annotation, 'segmented')
                    segmented.text = '0'
                    obj = ET.SubElement(annotation, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'Queenie Rose Amargo'
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '1'
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin_elem = ET.SubElement(bndbox, 'xmin')
                    xmin_elem.text = str(xmin)
                    ymin_elem = ET.SubElement(bndbox, 'ymin')
                    ymin_elem.text = str(ymin)
                    xmax_elem = ET.SubElement(bndbox, 'xmax')
                    xmax_elem.text = str(xmax)
                    ymax_elem = ET.SubElement(bndbox, 'ymax')
                    ymax_elem.text = str(ymax)
                    #For Saving Annotation
                    xml_filename = f"{current_time} {object_name}.xml"
                    xml_path = os.path.join(Queenie_Folder, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    print("Annotation XML file saved!")

            if object_name == "Reu Pan":
                face_detected_count += 1

                if face_detected_count == 20:
                    object_name = labels[int(classes[i])] 
                    label = '%s: DateTime: %s' % (object_name, date_time) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                    label_ymin = max(ymin, labelSize[1] + 10) 
                    current_time = time.strftime("%Y-%m-%d %H-%M-%S")
                    #For Saving Annotation
                    image_name = f"{current_time} {object_name}.jpg"
                    image_path = os.path.join(LJ_Folder, image_name)
                    cv2.imwrite(image_path, frame) 
                    print("Image captured and saved!")
                    face_detected_count = 0
                    captured_Reu_Pan = True
                    #For Annotation
                    annotation = ET.Element('annotation')
                    folder = ET.SubElement(annotation, 'folder')
                    folder.text = 'Face-Detected'
                    filename = ET.SubElement(annotation, 'filename')
                    filename.text = image_name
                    path = ET.SubElement(annotation, 'path')
                    path.text = os.path.abspath(image_path)
                    source = ET.SubElement(annotation, 'source')
                    database = ET.SubElement(source, 'database')
                    database.text = 'Unknown'
                    size = ET.SubElement(annotation, 'size')
                    width_elem = ET.SubElement(size, 'width')
                    width_elem.text = str(320)
                    height_elem = ET.SubElement(size, 'height')
                    height_elem.text = str(320)
                    depth = ET.SubElement(size, 'depth')
                    depth.text = str(3)
                    segmented = ET.SubElement(annotation, 'segmented')
                    segmented.text = '0'
                    obj = ET.SubElement(annotation, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'Reu Pan'
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '1'
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin_elem = ET.SubElement(bndbox, 'xmin')
                    xmin_elem.text = str(xmin)
                    ymin_elem = ET.SubElement(bndbox, 'ymin')
                    ymin_elem.text = str(ymin)
                    xmax_elem = ET.SubElement(bndbox, 'xmax')
                    xmax_elem.text = str(xmax)
                    ymax_elem = ET.SubElement(bndbox, 'ymax')
                    ymax_elem.text = str(ymax)
                    #For Saving Annotation
                    xml_filename = f"{current_time} {object_name}.xml"
                    xml_path = os.path.join(LJ_Folder, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    print("Annotation XML file saved!")

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Facial Recognition Dataset Automation', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
