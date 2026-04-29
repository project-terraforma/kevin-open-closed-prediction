# kevin-open-closed-prediction
**Author:** Kevin Nguyen

Project C: Open &amp; Closed Prediction

---

## Introduction
<!-- Will add a more formal explanation later -->
I noticed that several of the repos related to this project seemed to struggle with or did not highlight detection for closed places, so I wanted to make my contribution be about accuracy for closed places via precision, recall, and F1 score.

## OKRs
Objective 1: Improve how well the model detects closed places
* KR1.1: Closed precision, recall, and F1 score is ≥ 0.7 during testing
* KR1.2: Difference between closed precision and recall is ≤ 0.1

Objective 2: Make the model reliably detect closed places across different data sources
* KR2.1: Closed F1 score stays within ±0.05 across at least 2 different data sources
* KR2.2: Closed F1 score gap between training and testing is ≤ 0.1
​
## Findings

Here I will log all notes in regards to how the model behaves during training, and what affects its behavior towards accuracy for closed places and how.

**Logistic regression model:**
* For closed accuracy, turning on class balance creates low precision (many false positives), while turning off class imbalance low recall (many false negatives). Open accuracy is generally good, above 80%. In this case, threshold tuning matter significantly in changing the F1 score.
* The provided features from `project_c_samples.parquet` don't separate open and closed places strongly enough. Much more data than this is needed, as it's likely too hard to separate between open and closed as is now. Adding additional features does not strongly affect the F1 score, and instead simplification tends to lead to better scores.
* Numeric features (ex: confidence, contacts) are not as significant to the model as names or categories.

---

Created in association with the Overture Maps Foundation & CRWN 102 @ UCSC.