# kevin-open-closed-prediction

**Author:** Kevin Nguyen  
Project C: Open & Closed Prediction  

---

## Overview

This project explores whether a place is open or closed using Overture Maps place data. This repository's focus is on how well a model can detect the closed class, and what makes models struggle to achieve this, since closed places are the harder case and are often less clearly discussed in related work.

The work in this repository is a baseline logistic regression model built on the provided `project_c_samples.parquet` dataset. I experimented with text features, numeric features, class balancing, threshold tuning, and feature simplification to see what most affected closed-class performance.

## Goal

The goal of this project was to improve closed-place prediction and measure performance using class-specific metrics such as precision, recall, and F1 score.

## Repository Contents

This repository contains:
- the main training and evaluation script
- feature engineering code
- model evaluation using classification report and confusion matrix
- threshold tuning for closed-class performance


## Dataset

The main dataset used in this repository is:

- `project_c_samples.parquet`

This file contains pre-labeled place records with the target column:

- `open`  
  - `1` = place is open  
  - `0` = place is closed

Useful columns include:
- `names`
- `categories`
- `confidence`
- `websites`
- `socials`
- `phones`
- `brand`
- `addresses`
- `open`

---

## Approach

I used a logistic regression model with a mix of:

- text features from place names and categories
- structured categorical features
- numeric features based on contact presence and metadata
- class balancing
- threshold tuning for closed-class performance

I also tested several feature combinations to see whether additional engineered features improved closed prediction.

---

## Findings

### Logistic regression behavior
- Turning on class balancing improved closed recall, but it also increased false positives and lowered precision.
- Turning off class balancing improved precision, but closed recall dropped sharply.
- Threshold tuning had a noticeable effect on the closed F1 score.

### Feature behavior
- Name and category text were more useful than most numeric features.
- Numeric metadata such as confidence and contact information had limited predictive power on their own.
- Adding more engineered features did not strongly improve the model after a point.
- Simplifying the feature set often worked as well as or better than adding more features.

### Overall takeaway
- The provided data contains some signal for open/closed prediction, but closed detection remains difficult.
- The model performs much better on open places than closed places.
- Closed-class performance is sensitive to class imbalance and threshold choice.

---

## Key Insights

- **Text features carried most of the signal.**  
  Business names and categories were much more useful than most numeric metadata. TF-IDF features consistently dominated the model’s strongest coefficients.

- **Many engineered numeric features were redundant.**  
  Features like contact counts, missingness flags, and presence scores often described the same underlying signal, so adding more of them did not noticeably improve performance.

- **Class imbalance was the main challenge.**  
  The dataset is heavily skewed toward open places, so the model could achieve high overall accuracy while still doing poorly on closed-class recall and F1.

- **Threshold tuning mattered more than expected.**  
  Changing the decision threshold had a clear effect on closed-place performance, showing that the default 0.5 threshold was not ideal for this task.

- **Feature engineering reached diminishing returns.**  
  After several rounds of adding interaction features and other engineered signals, performance improvements became very small. This suggests that the current linear model has limited room to improve on this dataset alone.

- **The model learned meaningful business-type patterns.**  
  Certain names and categories consistently appeared as strong predictors, showing that the model was capturing real structure in the data rather than random noise.

- **The problem appears harder than a simple metadata task.**  
  Predicting whether a place is closed seems to require stronger or more time-sensitive signals than the provided static features capture.

---

## Future Work

Several directions could improve closed-place prediction beyond the baseline explored in this repository:

- **Tree-based models such as XGBoost or LightGBM**  
  These models may capture nonlinear relationships between metadata features better than logistic regression.

- **Temporal features**  
  Information such as edit history, update frequency, or time since last verification could provide stronger signals for whether a place is still active.

- **Geographic signals**  
  Regional patterns, population density, or nearby business activity may help distinguish between stable and unstable locations.

- **Richer business metadata**  
  Additional attributes such as hours consistency, review activity, or external business information may improve separation between open and closed places.

- **Embeddings instead of TF-IDF**  
  Modern embedding approaches could capture semantic meaning in names and categories more effectively than sparse TF-IDF features.

- **Improved imbalance handling**  
  Techniques such as resampling, focal loss, or alternative optimization strategies may improve closed-class recall without heavily sacrificing precision.

- **Ensemble methods**  
  Combining multiple models may improve robustness and overall closed-class performance.

- **Larger and more diverse datasets**  
  Additional training data may help the model generalize better and learn stronger patterns for closed-place prediction.

---

## Notes

This project is meant as a research baseline rather than a final production model. The main contribution is understanding what affects closed-place prediction and what does not.

---

## Acknowledgements

Created in association with the Overture Maps Foundation and CRWN 102 @ UCSC.