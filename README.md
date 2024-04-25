# AML24MST
Mini Project: Music Segmentation Task 

## Installation
Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate #Mac/Linux
.\venv\Scripts\activate.bat #Windows
pip install -r requirements.txt 
```

## Get started 

Step 1: Download the GTZAN dataset from Kaggle using the following link:
[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Save the dataset in a new folder *.data/* 

Step 2: Create train/test split data, and noise data by running the following:
```bash
python create_train_test_data.py
python3 create_noise_data.py
```