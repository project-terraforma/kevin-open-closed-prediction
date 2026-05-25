# kevin-open-closed-prediction

**Author:** Kevin Nguyen  
Project C: Open & Closed Prediction  

---

## Overview

This project explores whether a place is open or closed using Overture Maps place data. The primary focus of this repository is understanding how well a model can detect the closed class and what factors make closed-place prediction difficult, since closed places are both less common and more difficult to classify accurately.

This repository implements a baseline logistic regression model trained on the provided `project_c_samples.parquet` dataset. The project experiments with text features, numeric metadata, class balancing, threshold tuning, and feature simplification to analyze what most affects closed-class performance.

---

## Goal

The goal of this project was to improve closed-place prediction and evaluate performance using class-specific metrics such as precision, recall, and F1 score, with particular emphasis on the closed class.

---

## Repository Contents

This repository contains:
- the main training and evaluation script
- feature engineering code
- model evaluation using classification reports and confusion matrices
- threshold tuning for closed-class performance
- automatic saving of metrics and feature outputs
- generated visualizations for model analysis

The training script generates visualization outputs for:
- class distribution
- confusion matrix
- top feature coefficients

---

## Dataset

The primary dataset used in this repository is:

- `project_c_samples.parquet`

This dataset contains pre-labeled place records with the target column:

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

## Repository Structure

```text
kevin-open-closed-prediction/
│
├── data/
│   └── project_c_samples.parquet
│
├── results/
│   ├── final_metrics.txt
│   ├── top_features.txt
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── top_features.png
│
├── train_open_closed.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Approach

The model uses logistic regression with a combination of:

- TF-IDF text features from business names and categories
- structured categorical features
- engineered numeric metadata features
- class balancing
- threshold tuning for closed-class optimization

Additional feature engineering experiments were also tested to determine whether more complex engineered signals improved closed-place prediction performance.

---

## Results

Using a tuned logistic regression baseline with TF-IDF text features and engineered metadata features:

- Closed-class F1 score reached approximately **0.29**
- Open-class F1 score reached approximately **0.94**
- Macro F1 score reached approximately **0.62**
- Threshold tuning improved closed-place detection compared to the default threshold

Generated outputs include:
- classification reports
- confusion matrices
- top learned feature coefficients
- dataset distribution visualizations

**Note:** The primary contribution of this repository is not achieving the highest possible prediction accuracy. Instead, the project focuses on understanding why closed-place prediction is difficult and identifying which modeling decisions most strongly affect closed-class performance.

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

## Limitations

This project is meant as a research baseline rather than a final production model. The main contribution is understanding what affects closed-place prediction and what does not.

---

## Running the Project

### Prerequisites
- Python 3.10+
- pip package manager

### Steps
1. **Clone the repository**

    `git clone https://github.com/project-terraforma/kevin-open-closed-prediction.git`

2. **Create a virtual environment (optional but recommended)**
    ```
    python -m venv venv
    source venv/bin/activate    # Linux/Mac
    venv\Scripts\activate       # Windows
    ```

3. **Install dependences**

    `pip install -r requirements.txt`

4. **Run the training script**

    `python train_open_closed.py`

Results and visualizations will automatically be saved to the `results/` directory.

---

## Acknowledgements

Created in association with the Overture Maps Foundation and CRWN 102 @ UCSC.