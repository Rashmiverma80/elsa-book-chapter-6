AI-Based Structural Health Monitoring of Digital Concrete
CNN-Based Crack Detection using SDNET2018 Dataset
1. Introduction
This repository contains a Python implementation of a Convolutional Neural Network (CNN) for automated crack detection in concrete surfaces. The model is trained and evaluated using the SDNET2018 benchmark dataset, which consists of real-world images of cracked and non-cracked concrete surfaces collected from civil infrastructure.
The code supports the complete workflow described in the book chapter “AI in Structural Monitoring of Digital Concrete: Smart Solutions for Longevity”, including data preprocessing, model development, training, and performance evaluation using a confusion matrix and ROC curve.
2. Dataset Information
Dataset Used
SDNET2018 – Concrete Crack Image Dataset
•	Type: Image-based dataset
•	Total images: ~56,000
•	Classes:
o	Cracked concrete surfaces
o	Non-cracked concrete surfaces
•	Image source: Bridge decks, pavements, and walls
•	Data format: Grayscale images
Official Dataset Link
https://doi.org/10.17632/5y9wdsg2zt.2 
Note: This dataset is widely used in the literature for CNN-based crack detection and serves as a benchmark for structural health monitoring research.
3. Dataset Folder Structure
After downloading and extracting the dataset, organize it as follows:
SDNET2018/
├── Crack/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── NonCrack/
    ├── image_101.jpg
    ├── image_102.jpg
    └── ...
Ensure that:
•	Folder names are exactly Crack and NonCrack
•	Images are in .jpg or .png format
•	The dataset directory is placed in the same location as the Python script
4. System Requirements
Hardware
•	CPU: Any modern multi-core processor
•	RAM: Minimum 8 GB (16 GB recommended)
•	GPU (optional): NVIDIA GPU with CUDA support for faster training
Software
•	Operating System: Windows / Linux / macOS
•	Python version: Python 3.8 – 3.11
5. Environment Setup
5.1 Create a Virtual Environment (Recommended)
python -m venv cnn_shm_env
Activate the environment:
Windows
cnn_shm_env\Scripts\activate
Linux / macOS
source cnn_shm_env/bin/activate
5.2 Install Required Libraries
Install all dependencies using pip:
pip install numpy opencv-python matplotlib scikit-learn tensorflow
TensorFlow will automatically use the GPU if a compatible CUDA-enabled device is available.
6. Code Description
The Python script performs the following steps:
Section	Description
Import Libraries	Loads required Python packages
Dataset Loading	Reads SDNET2018 images and assigns labels
Preprocessing	Resizes, normalizes, and reshapes images
Model Development	Defines CNN architecture
Training	Trains CNN with validation split
Prediction	Generates class probabilities
Evaluation	Computes confusion matrix and ROC–AUC
7. Running the Code
1.	Place the Python script (e.g., cnn_crack_detection.py) in the same directory as the SDNET2018 folder
2.	Activate the virtual environment
3.	Run the script:
python cnn_crack_detection.py
8. Output Results
After execution, the following outputs will be displayed:
8.1 Confusion Matrix
•	Shows classification results for cracked and non-cracked surfaces
•	Visualized using a blue color map
8.2 ROC Curve
•	Displays True Positive Rate vs False Positive Rate
•	Includes ROC–AUC score indicating model performance
These results correspond to the combined performance evaluation figure used in the Results & Discussion section of the book chapter.
9. Model Configuration
Parameter	Value
Image size	128 × 128
Batch size	32
Epochs	10
Optimizer	Adam
Learning rate	0.0001
Loss function	Categorical Cross-Entropy
Output classes	Crack / Non-Crack
10. Reproducibility Notes
•	Train–test split uses a fixed random seed (random_state = 42)
•	Results may vary slightly depending on:
o	Hardware configuration
o	TensorFlow version
o	Number of training epochs
For improved performance, users may increase training epochs or apply data augmentation.
11. Citation
If you use this dataset or methodology in academic work, please cite:
Dorafshan, S., Thomas, R. J., & Maguire, M. (2018). SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. Data in Brief, Elsevier.
12. Intended Use
This implementation is intended for:
•	Academic research
•	Structural health monitoring studies
•	AI-based analysis of digital concrete
•	Educational demonstrations of CNN-based crack detection

# elsa-book-chapter-6
