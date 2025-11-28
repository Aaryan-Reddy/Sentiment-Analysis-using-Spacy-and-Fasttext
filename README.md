# Sentiment Analysis on Twitter Game Reviews

## Overview
This project performs sentiment analysis on Twitter data related to video games using Natural Language Processing (NLP) techniques. The pipeline covers data loading, preprocessing with spaCy and FastText, feature extraction using word embeddings, and training a machine-learning model (Random Forest) to classify tweets into sentiment categories (Positive, Negative, Neutral, Irrelevant).

## Dataset
- File: `twitter_training.csv` loaded with pandas.
- Columns: `Game_Name`, `Sentiment`, `tweet`.  
- Sentiment labels are roughly balanced across four classes.

## Method
1. Preprocess tweet text with spaCy (tokenization, normalization). 
2. Generate dense embeddings using both spaCy vectors and FastText word vectors, aggregated per tweet.  
3. Train `RandomForestClassifier(n_estimators=200, random_state=42)` on the combined embedding features to predict sentiment.

## Usage
1. Install dependencies:
pip install pandas numpy spacy scikit-learn fasttext
python -m spacy download en_core_web_sm
2. Place `twitter_training.csv` next to `SentimentAnalysisNLP.ipynb`. 
3. Open the notebook and run all cells to train and evaluate the model.
