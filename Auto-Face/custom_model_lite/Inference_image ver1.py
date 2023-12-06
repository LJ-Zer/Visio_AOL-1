import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import xml.etree.ElementTree as ET

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
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    required=True)
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder',
                    action='store_true')
parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)',
                    action='store_false')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu
save_results = args.save_results  # Defaults to False
show_results = args.noshow_results  # Defaults to True
IM_DIR = args.imagedir


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
    # If the user has specified the name of the .tflite file, use that name, otherwise use the default 'edgetpu.tflite'
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to the current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp') + glob.glob(PATH_TO_IMAGES + '/*.JPEG')

# Create results directory if the user wants to save results
if save_results:
    RESULTS_DIR = IM_DIR + '_results'
    RESULTS_PATH = os.path.join(CWD_PATH, RESULTS_DIR)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del labels[0]
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

save_folder_Leo = 'Face-Detected-Leo'
if not os.path.exists(save_folder_Leo):
    os.makedirs(save_folder_Leo)

save_folder_LJ = 'Face-Detected-LJ'
if not os.path.exists(save_folder_LJ):
    os.makedirs(save_folder_LJ)

save_folder_Kiko = 'Face-Detected-Kiko'
if not os.path.exists(save_folder_Kiko):
    os.makedirs(save_folder_Kiko)

save_folder_Queenie = 'Face-Detected-Queenie'
if not os.path.exists(save_folder_Queenie):
    os.makedirs(save_folder_Queenie)

save_folder_Reu = 'Face-Detected-Reu'
if not os.path.exists(save_folder_Reu):
    os.makedirs(save_folder_Reu)

leo_delen_counter = 0
lord_john_perucho_counter = 0
queenie_rose_amargo_counter = 0
frank_lester_castillo_counter = 0 
reu_pan_counter = 0 

outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:  #
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

############################################################################################################ Start ##############################################################################
for image_path in images:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if 0 <= int(classes[i]) < len(labels) and (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            object_name = labels[int(classes[i])]  # Look up object name from the "labels" array using the class index

    # Loop over every image and perform detection
    for image_path in images:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (int(width), int(height)))  # Convert width and height to integers
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

        for i in range(len(scores)):
            if 0 <= int(classes[i]) < len(labels) and (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                object_name = labels[int(classes[i])]  # Look up object name from the "labels" array using the class index

                if object_name == "Leo Delen":
 
                    # Save the resized cropped image
                    image_name = f"{object_name} ({leo_delen_counter}).jpg"
                    image_path = os.path.join(save_folder_Leo, image_name)
                    cv2.imwrite(image_path, image) 
                    print("Resized and cropped image captured and saved!")


                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

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

                    xml_filename = f"{object_name} ({leo_delen_counter}).xml"
                    xml_path = os.path.join(save_folder_Leo, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    leo_delen_counter += 1
                    print("Annotation XML file saved!")


                if object_name == "Lord John Perucho":

                    # Save the resized cropped image
                    image_name = f"{object_name} ({lord_john_perucho_counter}).jpg"
                    image_path = os.path.join(save_folder_LJ, image_name)
                    cv2.imwrite(image_path, image) 
                    print("Resized and cropped image captured and saved!")
                    
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # For Annotations
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

                    xml_filename = f"{object_name} ({lord_john_perucho_counter}).xml"
                    xml_path = os.path.join(save_folder_LJ, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    lord_john_perucho_counter += 1
                    print("Annotation XML file saved!")

                if object_name == "Frank Lester castillo":

                    # Save the resized cropped image
                    image_name = f"{object_name} ({frank_lester_castillo_counter}).jpg"
                    image_path = os.path.join(save_folder_Kiko, image_name)
                    cv2.imwrite(image_path, image) 
                    print("Resized and cropped image captured and saved!")

                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # For Annotations
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
                    name.text = 'Frank Lester Castillo'
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

                    xml_filename = f"{object_name} ({frank_lester_castillo_counter}).xml"
                    xml_path = os.path.join(save_folder_Kiko, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    frank_lester_castillo_counter += 1
                    print("Annotation XML file saved!")

                if object_name == "Queenie Rose Amargo":

                    # Save the resized cropped image
                    image_name = f"{object_name} ({queenie_rose_amargo_counter} s).jpg"
                    image_path = os.path.join(save_folder_Queenie, image_name)
                    cv2.imwrite(image_path, image) 
                    print("Resized and cropped image captured and saved!")
                    
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # For Annotations
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

                    xml_filename = f"{object_name} ({queenie_rose_amargo_counter} s).xml"
                    xml_path = os.path.join(save_folder_Queenie, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    queenie_rose_amargo_counter += 1
                    print("Annotation XML file saved!")

                if object_name == "Reu Pan":

                    # Save the resized cropped image
                    image_name = f"{object_name} ({reu_pan_counter}).jpg"
                    image_path = os.path.join(save_folder_Reu, image_name)
                    cv2.imwrite(image_path, image) 
                    print("Resized and cropped image captured and saved!")
                    
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # For Annotations
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

                    xml_filename = f"{object_name} ({reu_pan_counter}).xml"
                    xml_path = os.path.join(save_folder_Reu, xml_filename)
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path)
                    reu_pan_counter += 1
                    print("Annotation XML file saved!")

    if show_results:
        cv2.imshow('Face Detection', image)

        # Press any key to continue to the next image, or press 'q' to quit
        if cv2.waitKey(0) == ord('q'):
            break

cv2.destroyAllWindows()