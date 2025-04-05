# BTC Price & Sentiment Analysis with Deep Learning and Signature-Based Features

This repository is to analyze Bitcoin (BTC) price data with contextual data, especially sentimental information extracted and preprocessed from twitter(X). It integrates several time-series deep learning models for sentiment classification , as well as advanced time-series feature extraction (signature path) to predict BTC average prices.

## Overview

- **Price Data Acquisition**: 
  - Fetch BTC historical price data via Yahoo Finance.
  - Compute daily and weekly returns and save the processed data.

- **Sentiment Classification Model Training**:
  - Load a tweet corpus and preprocess the text data using tokenization and padding.
  - Convert sentiment labels to numerical format.
  - Build and train deep learning models:
    - LSTM
    - GRU
    - Simple RNN
    - Transformer (decoder-only)
  - Compare model performance through loss and accuracy plots.

- **BTC Twitter Analysis**:
  - Use the trained model to predict sentiment polarity on scraped BTC tweets with timestamp.
  - Compute daily average sentiment scores and merge with BTC return data.
  - Visualize weekly and daily trends in BTC returns and sentiment to show the correlation between sentiment and BTC price, and further more prove the predictability of BTC price by twitters.

- **Signature-Based Feature Integration & Regression**:
  - Extract signature features from BTC prices using the `esig` package.
  - Prepare features using sliding windows, lag/lead transformations, and time encoding.
  - Predict the average BTC price using:
    - Lasso Regression (with and without contextual features)
    - XGBoost Regression (with and without contextual features)
  - Tune model parameters with GridSearchCV and evaluate predictions using MAE and MAPE.
  - Generate dynamic plots to visualize model predictions.

## Requirements

- **Python 3.x**
- **Libraries**:
   - run `pip install -r requirements.txt` to install the required python packages.

## Data Files

- **BTC Data**: Automatically downloaded via `yfinance` for the period `2022-03-07` to `2023-03-06`.
- **Tweet Corpus**: `data/corpus.csv` (for training sentiment models).
- **Scraped Tweets**: `data/scraped_twitter.csv` (for BTC sentiment analysis).
- **Pre-trained Embeddings**: `data/glove.twitter.27B.100d.txt` (used in model embedding layers).