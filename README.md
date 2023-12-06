# Automation-of-Labelling

## INITIAL PHASE

1. Install Anaconda
2. Create Virtual Environment
	- conda create --name automatic python=3.9
3. Activate the Environment
	- conda activate automatic
4. Install Dependencies
	- pip install tensorflow opencv-python protobuf==3.20.*

## INFERENCING PHASE

1. Activate the Environment
	- conda activate automatic
2. Run the Python for Image
   * Run the script in Auto-Face directory. Make sure to change the imagedir.
	- python custom_model_lite/Inference_image ver1.py --imagedir "path/of/your/images" --modeldir=custom_model_lite
3. Run the Python for Real-time Dataset Fetcher
	- python custom_model_lite/Inference_webcam ver1.py --modeldir=custom_model_lite

Credit to all of the owners of the default Python Scripts. This is used for educational purposes. 
