# AML24MST
This is the reposirory for the mini-project in Advanced Machine Learning 2024 at IT-University of Copenhagen.

## Group Members
Anders Hjulmand: ahju@itu.dk

Eisuke Okuda: eiok@itu.dk  

Andreas Flensted: frao@itu.dk

## Data and Objective
The objective of this project is to build a classifier that can tag pieces of music with a genre. 

We use the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) which is the most-used public dataset for evaluating music genre recognition in machine learning. The dataset consists of 10 genres with 100 audio files of 30 second each. For each genre, the data is split into training (60), validation (20), and test (20).

The main goals of the project is to:

1. Compare the performance of different pre-processing options (raw waveform vs. spectograms). 
2. Compare different model architectures: spectogram CNN vs. transformers.
3. Examine whether including noisy- and generated data as training examples improves the performance of the models.


Spectograms of a single song from the 10 genres are shown in the figure below. 
![](figures/ast_spectograms.png)


## Methods

We fine-tune three pre-trained model architectures to classify music genres in the GTZAN:

1. CNN - [Paper](url): 
2. Audio Spectogram Transformer (AST) - [Paper](https://arxiv.org/abs/2104.01778): which is an attention-based vision transformer model used for audio classification. The audio is first turned into a spectogram, then projected onto an embedding space, after which a vision transformer is applied. We use a model pretrained on the [AudioSet](https://research.google.com/audioset/) consisting on a variety of audio classes including music and speech. This corresponds to the 1st model on the [GitHub from ](https://github.com/YuanGongND/ast/tree/master?tab=readme-ov-file). A learning rate of 5e-5 for fine-tuning. 
4. Hubert - [Paper](https://arxiv.org/abs/2106.07447): 


## Key Experiments and Results

We examine whether augmenting the training data with noisy and generated music pieces improves the performance. 

We create noisy training data by adding Additive White Gaussian Noise (AWGN) with a signal-to-noise-ratio of $10$ to the original training examples.

We generate ...  


Accuracy scores are shown in the table below.  

|          | ResNet | AST | HuBERT |
|----------|----------|----------|----------|
| Baseline data             | .692   | .775   | .795   |
| Baseline + Noise data     | .705   | .740   | .780   |
| Baseline + Generated data | .686   | .755   | .795   |


The figure below shows the confusion Matrix for the AST model fine-tuned on baseline data.
![](figures/confusion_plot_baseline_ast.png)


The figure below plots the first two principal components of all the songs in the original dataset. Large points indicates test-set songs that were misclassified by all 9 models. 
![](figures/pca_plot_misclassified.jpg)


## Discussion

In relation to the main goals we conclude the following:

1. The model that uses raw waveforms as input (HuBERT) has overall better accuracy than the models that uses mel-spectograms (AST and ResNet).
2. The transformer models (HuBERT and AST) have better accuracy than the CNN model (ResNet).
3. For the ResNet model, the adding noise data improces accuracy slightly (+ .013 from baseline). Neither noisy- or generated data improved the performance of the AST. Generated data archieved the same accuracy as baseline data in the HuBERT model. 

There was a big difference in the performance across genres. Classical- and jazz music was easier for the models to classify correctly, whereas disco and rock genres were more often misclassified. 

From the PCA-figure we note that the misclassified songs were generally placed in regions of the latent space with a lot of overlap between genres. The clusters that were more distinct (e.g. classical and metal) had no misclassifications. 

While listening to some of the songs that were misclassified by all the models we realized that we could not even agree on the genre of the song. See for example [this rock song](https://jumpshare.com/s/VVWPKtGIc0Pn8y5wtkth). Music genres are not "hard labels" like cats or dogs, but are ambigious and might depend on the knowledge and the musical taste of the listener.


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

Step 3: Create generated data by running the notebook MusicGen.ipynb.

## Fine-tune models

After creating the datasets, the following notebooks can br run to fine-tune the models:

* transfer_CNN.ipynb
* finetune_ast.ipynb
* hubert.ipynb





