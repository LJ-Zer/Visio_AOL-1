INITIAL PHASE

1. Install Anaconda
2. Create Virtual Environment
	- conda create --name automatic python=3.9
3. Activate the Environment
	conda activate automatic
4. Install Dependencies
	- pip install tensorflow opencv-python protobuf==3.20.*

INFERENCING PHASE

1. Activate the Environment
	- conda activate automatic
2. Run the Python
	- python custom_model_lite/TFLite_detection_image.py --imagedir "path/of/your/images" --modeldir=custom_model_lite  

