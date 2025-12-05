# Project 4: Twitter Sentiment Analysis

## Project Title: Twitter Sentiment Analysis — Crowdflower Dataset

### Group Members
- Catherine
- Tanveer
- Kelvin
- Cindy
- James

## 1.0 Introduction
This project analyzes Twitter feedback on Apple and Google products using Natural Language Processing (NLP). The aim is to classify tweets as Positive, Neutral, or Negative automatically, providing insights for brand monitoring, customer support, and product development.

## 2.0 Research Problem & Objectives

*Problem Statement:*  
Apple and Google products generate thousands of tweets daily, reflecting customer experiences, frustrations, and praise. Manually analyzing this volume is slow, inconsistent, and difficult to scale. The goal is to build an automated system for sentiment classification.

*Main Objective:*  
Develop a robust multi-class NLP model that classifies tweet sentiments to support data-driven decisions.

*Specific Objectives:*  
- Prepare and explore the dataset: Clean and preprocess tweets, analyze sentiment distribution.  
- Build and compare models: Train Logistic Regression, SVM, Random Forest, and LSTM models.  
- Optimize and evaluate performance: Apply pipelines, hyperparameter tuning, and evaluate using accuracy, F1-score, and confusion matrices.  
- Interpret and save models: Use LIME or SHAP for explainability and save the best model.  
- Provide actionable recommendations: Generate insights to guide marketing strategies, customer support, and product development.

*Stakeholders:*  
- Marketing Team: Monitor brand perception.  
- Customer Support: Detect negative feedback and prioritize complaints.  
- Product Team: Understand user reactions for feature development.  
- Data Science Team: Build, validate, and interpret models.  
- Business Executives: Use insights for strategic decisions.

*Project Scope:*  
- *In-Scope:* Using existing labeled tweets, model development, evaluation, interpretation, and saving models.  
- *Out-of-Scope:* Live data collection, real-time deployment, other brands, or advanced NLP techniques beyond LSTM.

## 3.0 Data Understanding

### 3.1 Dataset Overview
The dataset contains ~9,000 tweets from CrowdFlower/data.world, labeled as positive, negative, or neutral by human annotators.

### 3.2 Loading the Data
```python
import pandas as pd
df = pd.read_csv("crowdflower_twitter_sentiment.csv", encoding='latin1')
df.head()
### 3.3 Variable Description
- tweet_text: Raw tweet content.  
- emotion_in_tweet_is_directed_at: Brand/product mentioned (nullable).  
- is_there_an_emotion_directed_at_a_brand_or_product: Sentiment label (Positive, Negative, Neutral, No emotion, I can’t tell).

## 4.0 Data Cleaning & Pre-processing

### 4.1 Handling Missing Values
Remove or fill nulls in relevant columns.

### 4.2 Feature Engineering
Created clean_text column with lowercased, punctuation- and stopword-removed, stemmed tweets.

### 4.3 Text Preprocessing (Tokenization, Lemmatization, Stopwords)
Used NLTK for stopwords and stemming; optional: spaCy for lemmatization.

## 5.0 Exploratory Data Analysis (EDA)

### 5.1 Sentiment Distribution
Most tweets are neutral ("No emotion toward brand or product"). Positive sentiment is second, negative is limited, "I can't tell" minimal.

### 5.2 Common Words per Sentiment
Frequent words include sxsw, mention, ipad, google, apple, iphone. Stopwords removal necessary.

### 5.3 Word Clouds (optional)
Visual representation of the most common words per sentiment category.

## 6.0 Modeling

### 6.1 TF-IDF + Random Forest
TF-IDF vectorization followed by Random Forest classifier showed ~64% accuracy.

### 6.2 SHAP Explainability
Used LIME for local explanations; Random Forest predictions aligned with human-labeled sentiments.

### 6.3 SVD Dimensionality Reduction
Applied TruncatedSVD to reduce TF-IDF dimensions for visualization and pipeline improvement.

### 6.4 LSTM Deep Learning Model
Implemented LSTM with tokenization and padding; achieved ~59% test accuracy. Class weighting applied for imbalanced classes.

## 7.0 Model Evaluation

### 7.1 Classification Reports
Logistic Regression, Linear SVM, Random Forest, and LSTM reports provided; Random Forest performed best on minority classes.

### 7.2 Confusion Matrices
Visualized misclassifications; majority class (neutral) dominates.

### 7.3 ROC Curves (optional)
Not included; optional for future work.

## 8.0 Explainability Section

### 8.1 SHAP Summary Plots
LIME provided feature importance for individual tweet predictions.

### 8.2 Mapping SVD Components → Words
SVD reduced TF-IDF dimensions; top components associated with key sentiment words.

### 8.3 Interpretation
Models identify relevant words contributing to sentiment; LSTM learns sequence patterns; Random Forest provides interpretable insights.

## 9.0 Final Recommendation on Best Model
Random Forest is recommended for interpretability and balanced performance across sentiment classes. LSTM may improve with more data or further tuning.

## 10.0 Conclusion
Automated sentiment analysis on tweets about Apple and Google products is feasible. Models provide actionable insights for marketing, customer support, and product strategy.

## 11.0 References
- CrowdFlower Twitter Sentiment Dataset, data.world  
- NLTK Documentation  
- scikit-learn Documentation  
- TensorFlow/Keras Documentation  
- LIME: Local Interpretable Model-Agnostic Explanations