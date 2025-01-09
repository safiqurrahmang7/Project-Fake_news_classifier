# Fake News Classifier  

## Overview  
The **Fake News Classifier** is a deep learning-based project aimed at combating the spread of misinformation by identifying and classifying news articles as genuine or fake. The project leverages advanced Natural Language Processing (NLP) techniques and deep learning models to provide an efficient and accurate solution for fake news detection.  

## Problem Statement  
The spread of fake news is a critical issue in the media and entertainment domain. This project aims to develop a reliable classifier that can distinguish fake news from real news, thus helping mitigate the risks of misinformation.  

## Features  
- Classification of news articles as "Fake" or "Real."  
- Utilizes advanced NLP techniques for text preprocessing and feature engineering.  
- Built on a robust deep learning model using LSTM architecture.  
- Provides predictions with a high F1-score for reliable results.  

## Dataset  
- **Name**: WELFake Dataset  
- **Size**: 72,134 news articles  
- **Labels**:  
  - `0` for Fake News  
  - `1` for Real News  

## Technical Details  
### Skills Used  
- **Programming Languages**: Python  
- **Frameworks**: TensorFlow, Keras, unittest, spacy, mlflow
- **Tools**: Jupyter Notebook, Vs code, google colab
- **NLP Techniques**: Tokenization, Lemmatization, Word Embeddings  

### Approach  
1. **Data Collection**: Sourced the WELFake dataset.  
2. **Data Preprocessing**:  
   - Handled NaN values.  
   - Preprocessed the  `text` column.  
   - Used tokenization and other NLP techniques.  
3. **Exploratory Data Analysis (EDA)**: Visualized trends and relationships in the dataset.  
4. **Feature Engineering**: Applied word tokenization, paddig sequences and word embeddings.  
5. **Model Development**:  
   - Developed an LSTM-based neural network.  
   - Set `max_length` for sequences to 1000 for model input.  
6. **Model Evaluation**: Achieved a high F1-score on the test dataset.  

## Results  
The model demonstrates high accuracy in classifying fake and real news, ensuring reliable performance in real-world applications.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/safiqurrahmang7/Project-Fake_news_classifier.git
   cd Project-Fake_news_classifier
