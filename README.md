# Machine Learning Assignment 2

## 1. Problem Statement
Build and evaluate multiple classification models on one public classification dataset, expose the trained models through an interactive Streamlit application, and make the complete project reproducible through GitHub.

This project predicts whether a breast mass is **malignant (1)** or **benign (0)** from 30 numeric features extracted from digitized fine-needle-aspirate images.

> **Academic-use notice:** This project is for coursework demonstration only and is not a medical diagnostic tool.

## 2. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**UCI dataset ID:** 17  
**DOI:** 10.24432/C5DW2B  
**Instances:** 569  
**Predictor features:** 30  
**Target:** `diagnosis` (`1 = malignant`, `0 = benign`)  
**Missing values:** None in the packaged dataset used here.

The repository includes `breast_cancer_wisconsin_diagnostic.csv`, a reproducible CSV copy generated from scikit-learn's packaged version of the UCI dataset.

## 3. GitHub Repository Link
Replace this placeholder after pushing the repository:

`https://github.com/bipulshahi/ml-assignment-2-bits`

## 4. Models Used
The assignment text says “all 6 models” but explicitly lists five algorithms. To avoid under-submission, this project implements all five named algorithms **plus SVM as an additional sixth model**:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. Support Vector Machine (additional sixth model)

All models use the same train/test split (`random_state=42`, stratified, 80/20).

## 5. Evaluation Metrics
For each model, the project calculates:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

For Precision, Recall and F1, the positive class is **malignant (1)**.

## 6. Model Comparison

| ML Model Name                           |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:----------------------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression                     |     0.9649 | 0.996  |      0.975  |   0.9286 | 0.9512 | 0.9245 |
| Decision Tree                           |     0.8772 | 0.9654 |      0.9118 |   0.7381 | 0.8158 | 0.7343 |
| K-Nearest Neighbors                     |     0.9561 | 0.9825 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes                    |     0.9211 | 0.9891 |      0.9231 |   0.8571 | 0.8889 | 0.8292 |
| Random Forest                           |     0.9737 | 0.9954 |      1      |   0.9286 | 0.963  | 0.9442 |
| Support Vector Machine (Additional 6th) |     0.9737 | 0.9947 |      1      |   0.9286 | 0.963  | 0.9442 |

## 7. Model-wise Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy=0.9649, AUC=0.9960, F1=0.9512, MCC=0.9245. Scaling supports stable linear classification and strong probability ranking. |
| Decision Tree | Accuracy=0.8772, AUC=0.9654, F1=0.8158, MCC=0.7343. Easy to interpret, but a single tree can be more sensitive to the training split. |
| K-Nearest Neighbors | Accuracy=0.9561, AUC=0.9825, F1=0.9383, MCC=0.9058. Scaling is essential because the algorithm is distance-based. |
| Gaussian Naive Bayes | Accuracy=0.9211, AUC=0.9891, F1=0.8889, MCC=0.8292. Fast probabilistic baseline; its independence assumption can limit performance when predictors are correlated. |
| Random Forest (Ensemble) | Accuracy=0.9737, AUC=0.9954, F1=0.9630, MCC=0.9442. Aggregating many trees reduces variance and usually improves robustness over a single decision tree. |
| Support Vector Machine (Additional 6th) | Accuracy=0.9737, AUC=0.9947, F1=0.9630, MCC=0.9442. RBF SVM captures non-linear boundaries; scaling is applied through a pipeline. |
| **Overall Winner** | **Random Forest**, selected using average rank across Accuracy, AUC, Precision, Recall, F1 and MCC, with AUC/MCC used as tie-breakers. |

## 8. Repository Structure

```text
ML_Assignment_2_GitHub_Repository/
├── app.py
├── train.py
├── train_models.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── breast_cancer_wisconsin_diagnostic.csv
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── random_forest.joblib
│   └── svm.joblib
└── results/
    └── model_metrics.csv
```

## 9. How to Reproduce the ML Results

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
python train.py
```

This retrains all models, recreates `test_data.csv`, saves the model files, and writes `results/model_metrics.csv`.

## 10. How to Run the Streamlit App Locally

```bash
streamlit run app.py
```

Then:
1. Upload `test_data.csv`.
2. Select a model from the dropdown.
3. Review Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Review the confusion matrix and classification report.
5. Optionally download predictions.

## 11. Streamlit Community Cloud
After pushing this repository to GitHub:
1. Open Streamlit Community Cloud.
2. Create a new app.
3. Select this repository and the `main` branch.
4. Set the entry point to `app.py`.
5. Deploy.

Add the live URL here after deployment:

`https://bipul-ml-assignment-2.streamlit.app/`

## 12. BITS Virtual Lab Requirement
Run the training notebook or `train.py` in the BITS Virtual Lab and capture **one screenshot** showing successful execution. The screenshot is submitted in the final PDF as required by the assignment.

## 13. Final Submission PDF
The assignment requires a **single PDF**, maintaining this order:
1. Clickable GitHub repository link
2. Clickable live Streamlit app link
3. BITS Virtual Lab execution screenshot
4. README content

## 14. Academic Integrity
Use this repository as your own working project: review the code, understand each step, make your own Git commits, and customize the README/UI before submission.
