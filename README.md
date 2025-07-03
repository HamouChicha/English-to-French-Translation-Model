# English to French Translator

## Overview

This project builds a translator from English to French using a Seq2Seq model with an Attention mechanism. The model is designed to process sentences in English and translate them into French.

## Dataset

The dataset used for training the model is sourced from Kaggle. It contains pairs of English and French sentences, allowing the model to learn the nuances of translation between the two languages.

## Files Included

- `Language_Translation(English_French).ipynb`: Jupyter Notebook containing the full analysis, preprocessing steps, model training,...
- `functions.py`: Python script with reusable helper functions used throughout the notebook.
- `eng_-french.csv`: Dataset containing English–French sentence pairs used for training and testing.
- `Attention_Translator.keras`: The saved trained model with attention mechanism. [Download here](https://drive.google.com/file/d/1Oyad4Eu3fyY-RbT3t51JaRK3iTlwwrGF/view?usp=sharing).
- `requirements.txt`: List of required Python libraries and their versions for running the project.


## Model Architecture

The translation model utilizes:
- **Seq2Seq Architecture**: A sequence-to-sequence architecture that includes an encoder to process the input sentence and a decoder to generate the translated output.
- **LSTM Cells**: Long Short-Term Memory (LSTM) units are used in both the encoder and decoder to capture long-range dependencies and sequence order in the data.
- **Attention Mechanism**: The attention mechanism enhances the model by allowing the decoder to dynamically focus on relevant parts of the input sequence at each decoding step, leading to improved translation accuracy and fluency.

## Requirements

- Python 3.11.9
- All required libraries are listed in the `requirements.txt` file.