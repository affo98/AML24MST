# AML24MST
Mini Project: Music Segmentation Task 

## Data and Objective

## Methods

## Key Experiments and Results

## Discussion

## Installation
Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate #Mac/Linux
.\venv\Scripts\activate.bat #Windows
pip install -r requirements.txt 
```

## Create Datasets

Step 1: Download the GTZAN dataset from Kaggle using the following link:
[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Save the dataset in a new folder *.data/* 

Step 2: Create train/test split data, and noise data by running the following:
```bash
python create_train_val_test_data.py
python create_noise_data.py
```

## Group Members
Anders Hjulmand: ahju@itu.dk
Eisuke Okuda: eiok@itu.dk  
Andreas Flensted: frao@itu.dk
