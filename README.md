# Kickstarter Success Prediction: A Real-World Binary Classification Project

## Overview

This project was developed as part of a university-level competition aimed at predicting the **success of Kickstarter crowdfunding campaigns** using a real-world dataset containing over 100,000 project records.

The binary classification task required participants to build a model that predicts whether a given Kickstarter project would successfully meet or exceed its fundraising goal (`YES`) or not (`NO`). The dataset includes both structured and unstructured features such as campaign text, images, goal amount, creator metadata, reward tiers, and campaign durations.

---

## Objective

- **Target Variable**: `success`  
  - `YES`: Project met or exceeded its funding goal  
  - `NO`: Project failed to meet the goal  

- **Goal**: Build a predictive model using machine learning techniques in **R** to classify test data with the highest possible accuracy.

---

## Approach

The project was implemented in **two phases**, evolving from a basic logistic regression model to an advanced XGBoost pipeline with comprehensive feature engineering and threshold tuning.

### Phase 1: Logistic Regression Baseline
- **Script**: `phase1`
- Focused on building a baseline logistic regression model using cleaned numeric, date, and text-derived features.
- Engineered a few key features such as:
  - Campaign duration
  - Preparation time
  - Blurb length
  - Reward amount average
- Achieved ~60% accuracy on test data using a threshold of 0.49.

### Phase 2: Final Model with XGBoost
- **Script**: `phase2`
- Implemented a robust machine learning pipeline with:
  - 100+ **engineered features**
  - **Correlation filtering** and **zero-variance removal**
  - **Parallelized XGBoost training** with early stopping
  - **Custom threshold tuning** based on validation accuracy
- Categories of features included:
  - Textual sentiment, entropy, spelling errors
  - Image presence and brightness
  - Launch timings, weekday/month patterns
  - Tag frequency and keyword presence
  - Reward tiers and goal normalization
- Achieved a final accuracy of **82.82%**, the highest score in the competition.


## Feature Engineering Highlights

- **Text Features**:
  - Entropy, sentiment scores (syuzhet)
  - Capitalization, length, spelling mistakes (hunspell)
- **Date Features**:
  - Campaign duration, prep time, weekday/month indicators
- **Reward Features**:
  - Tier count, min/max/avg values, keywords in descriptions
- **Goal Features**:
  - Normalized and log-transformed goal
  - Per-day goal, categorized goal size
- **Visual/Image Features**:
  - Color brightness from hex codes
  - Type counts (text, logo, calendar images)
- **NLP-Based POS Tags**:
  - Word/sentence-level syntactic structure using AFINN and part-of-speech tags
- **External Metadata**:
  - Currency, location, and scraped creator information

## Best Model Training and Evaluation

- Model: **XGBoost** (`binary:logistic`)
- Training set: ~85,000 instances  
- Test set: ~11,308 instances  
- Evaluation metric: **Accuracy**
- Threshold tuning: Evaluated 1,000+ thresholds to select the one with highest match to ground truth.
