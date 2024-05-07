# AML24MST
This is the reposirory for the mini-project in Advanced Machine Learning 2024 at IT-University of Copenhagen.

## Group Members
Anders Hjulmand: ahju@itu.dk

Eisuke Okuda: eiok@itu.dk  

Andreas Flensted: frao@itu.dk

## Data and Objective
The objective of this project is to build a classifier that can tag pieces of music with a genre. 

We use the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) that with its 1000 songs evenly spread across 10 different genres in both raw audio and Mel Spectrogram image format, is the most-used public dataset for evaluating music genre recognition. This project leverages the raw audio files each of 30 seconds duration. For each of the genres, the data is split into training (60), validation (20), and test (20) set - referenced as baseline data.

The main goals of the project is to:

1. Evaluate the performance of using raw waveform and spectrograms as pre-processing methods to represent the audio signal.
2. Evaluate the performance of different model architectures: ResNet vs. Transformers.
3. Evaluate the robustness of model performance by exposing noisy- and generated audio data as training examples.


Mel Spectrograms of a single song from the 10 genres are shown in the figure below. 
![](figures/ast_spectograms.png)

## Models

We fine-tune 3 pre-trained model architectures to classify music genres;

1. ResNet50 - [Paper](https://arxiv.org/abs/1512.03385): Finetuning the ResNet50 architecture trained on ImageNet to use for feature extraction as well as an inserted convolutional layer at the beginning and a Multi Layer Perceptron at the end. The models were trained using mel spectrograms with a learning rate of 1e-4 for 20 epochs with an early stopping criterion.
2. Audio Spectogram Transformer (AST) - [Paper](https://arxiv.org/abs/2104.01778): is an attention-based vision transformer model used for audio classification. The input to the model is a mel spectogram, that is projected onto an embedding space, after which a vision transformer encoder is applied. We use a model pretrained on the [AudioSet](https://research.google.com/audioset/) consisting on a variety of audio classes including music and speech. This corresponds to the 1st model on the [GitHub from the original paper](https://github.com/YuanGongND/ast/tree/master?tab=readme-ov-file). We finetuned the model using a learning rate of 5e-5 for 15 epochs and based the model selection on the validation accuracy. 
3. Hubert - [Paper](https://arxiv.org/abs/2106.07447): is a self-supervised approach for speech representation learning. HuBERT uses an offline clustering step to provide aligned target labels for a BERT-like prediction loss. The prediction loss is applied only over the masked regions, forcing the model to learn a combined acoustic and language model over the continuous inputs. The models were finetuned with a learning rate 5e-5 for 10 epochs.


*All models were trained using Kaggle Notebooks with 2x GPU T4, 4 CPU, and 16 GB RAM.*

## Data Augmentation

We examine whether augmenting the training data with noisy and generated music pieces improves the performance, or if any of the architectures performance's are sensitive to the exposure of noisy or generated data during training.

The noisy training data was created by adding Additive White Gaussian Noise (AWGN) with a signal-to-noise-ratio of $10$ to the original data. The noisy data was added to the train and validation sets which doubled the number of training examples. Noisy audio data was *not* added to the test set.

We generated 1000 artificial audios using [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1), to generate 50 prompts, which was then fed to TextToMusicGenerating [MusicGen](https://huggingface.co/spaces/facebook/MusicGen), which resulted in 800 additional training examples and 200 validation examples. Generated audio data was *not* added to the test set.


## Key Experiments and Results

The combination of 3 architechtures and 3 training-sets resulted in 9 different models.

Accuracy scores are shown in the table below.  

|          | ResNet | AST | HuBERT |
|----------|----------|----------|----------|
| Baseline data             | .692   | .775   | .795   |
| Baseline + Noise data     | .705   | .740   | .780   |
| Baseline + Generated data | .686   | .755   | .795   |


The figure below shows the confusion Matrix for the AST model fine-tuned on baseline data.
![](figures/confusion_plot_baseline_ast.png)


The figure below plots the first two principal components of all the songs in the baseline dataset using the features provided in the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Large points represent test set songs that were misclassified by all 9 models. We note that the misclassified songs were generally placed in regions of the latent space with a lot of overlap between genres. The clusters that were more distinct (e.g. classical and metal) had no misclassifications. 
![](figures/pca_plot_misclassified.jpg)


## Discussion

1. The model that uses raw waveforms as input (HuBERT) has overall better accuracy than the models that uses mel-spectrograms (AST and ResNet50).
2. The transformer models (HuBERT and AST) have better accuracy than the CNN model (ResNet50).
3. Adding noise data to the ResNet50 model improves accuracy slightly (+ .013 from baseline). Neither noisy- or generated data improved the performance of the AST. Generated data achieved the same accuracy as baseline data in the HuBERT model. None of the models had large decrease in accuracy when trained on noisy or generated audio files. This alludes to the models learning feature representations that are not only artifacts of the GTZAN data but could be applied to new test sets.

There was a big difference in the performance across genres. Classical- and jazz music was easier for the models to classify correctly, whereas disco and rock genres were more often misclassified. When adding noisy data to the training data of the ResNet50 architechture, the correctly classified rock songs increased from 1 to 12. This raises the question of whether supervised music tagging methods could benefit from genre based pipelines. 

While listening to some of the songs that were misclassified by all of the models we realized that we too could not classify the genre of the audio file. See for example [this rock song](https://jumpshare.com/s/VVWPKtGIc0Pn8y5wtkth). This states that music genres are not "hard labels" like cats or dogs, but are ambigious and might depend on the knowledge and the music taste of the listener.

To improve the evalution of the model performance, future work would leverage another music genre dataset to identify the impact of the noisy and generated training examples on unseen audio data.

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

After creating the datasets, the following notebooks can be run to fine-tune the models:

* GTZAN_RESNET50.ipynb
* finetune_ast.ipynb
* hubert.ipynb





