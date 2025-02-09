# Pancreatic_cancer_detection
Pancreatic ductal adenocarcinoma (PDAC) is a highly aggressive malignancy with the five-year survival rate of only 13% largely due to late-stage diagnosis. Accurate and early detection is critical for improving survival outcomes. 

Considering the importance and prevalence of pancreatic cancer worldwide, we trained an effective CNN-based multiclass classification model to classify computed tomography (CT) scan images of pancreas int three classes such as pancreatic cancer, pancreatitis and normal pancreas. The ResNet50-based CNN model achieved highest average accuracy of approximately 99.80% on the training set (100% precision and approximately 99% recall) and accuracies of 99.50% and 99.16% on the validation and additional test set. 

## Package requirements:
* python = 3.12 
* numpy
* pandas
* tensorflow= 2.18.0
* keras = 3.6.0
* scikit-learn
* scipy
    
## Repository files:
The files contained in this repository are as follows:
* ``Readme file``: General information
* Folder named ``Model`` contains the trained model
* ``prediction_pc.py``: Main script to run prediction
* Folder named ``Images``: Folder where images to be classified are to be saved (sample images are given for the reference).

## Usage:

> **_NOTE:_** Remember to activate the corresponding conda environment before running the script, if applicable.

**Input**: Image file (image.jpg)

**Output**: Prediction file

**Execution:**
**Step 1**: Download this repository, extract zipped file and ensure that all the files are present in the same folder when running the script.

**Step 2**: Install Anaconda3-5.2 or above.

**Step 3**: Before running these scripts, create a new conda environment and install the following packages using the pip command (pip install name_of_package = version)
            Install or upgrade libraries mentioned above (python, numpy, pandas, tensorflow, keras, scikit-learn, scipy).

**Step 4**: Save the images to be predicted in "Images" folder

**Step 5**: Execute the script ``prediction_pc.py``.

    python prediction_pc.py
            
The script runs in the current directory.

**Step 6**: Prediction results will be saved in `predictions.csv` which includes predicted output prediction class of pacreas CT images with probility of prediction.
