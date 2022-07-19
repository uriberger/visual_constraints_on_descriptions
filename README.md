# visual_constraints_on_descriptions

Installation:

1. Install requirements from requirements.txt file.
2. Run setup.csh with a path to a python executable.
3. git clone the Recognizers-Text github package (https://github.com/microsoft/Recognizers-Text.git) and follow the python installation steps in https://github.com/Microsoft/Recognizers-Text/tree/master/Python. Then, open the number_recognizer file (Recognizers-Text\Python\libraries\recognizers-number\recognizers_number\number\number_recognizer.py), go to the NumberRecognizer class, initialize_configuration function and search for the "# region German" mark. If this region is commented, uncomment it.
4. Download the MoCo pre-trained models from the following link: https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar, create a directory named 'models' under the project root and place the MoCo model in it.

Execution:
- To train models use the main.py file (run with --help to see all options)
- To run corpus analysis experiments use the data_analysis.py file
