# Random Forest From Scratch
Random Forest Algorithm written in Python using NumPy and Pandas. Based on the [Decision Tree project](https://github.com/huannpham/decision-tree-from-scratch).
## 1. Overview of the Implemention

## 2. Repository Structure
```
decision-tree-from-scratch/
├── dataset_files/
│   ├── cancer_X_train.csv  # Breast Cancer Wisconsin (Diagnostic) Training Dataset
│   ├── cancer_y_train.csv  # Breast Cancer Wisconsin (Diagnostic) Training Labels
│   ├── cancer_X_test.csv   # Breast Cancer Wisconsin (Diagnostic) Testing Dataset
│   ├── cancer_y_test.csv   # Breast Cancer Wisconsin (Diagnostic) Testing Labels
│   ├── car_X_train.csv     # Car Evaluation Training Dataset
│   ├── car_y_train.csv     # Car Evaluation Training Labels
│   ├── car_X_test.csv      # Car Evaluation Training Dataset
│   └── car_y_test.csv      # Car Evaluation Training Labels
│
├── decisionTree.py         # Decision Tree Algorithm
├── randomForest.py         # Random Forest Algorithm
├── breastCancer.py         # Training and Testing on Breast Cancer Wisconsin (Diagnostic) Dataset
└── carEvaluation.py        # Training and Testing on Car Evaluation Dataset
```
## 3. Testing Specifications
- 
- There is no preprocessing required for the UCI Breast Cancer Wisconsin (Diagnostic) Dataset.
- The UCI Car Evalution Dataset will be preprocessed as follows.
```
"buying": "low" -> 1, "med" -> 2, "high" -> 3, "vhigh" -> 4
"maint": "low" -> 1, "med" -> 2, "high" -> 3, "vhigh" -> 4
"doors": "2" -> 2, "3" -> 3, "4" -> 4, "5more" -> 5
"persons": "2" -> 2, "4" -> 4, "more" -> 6
"lug_boot": "small" -> 1, "med" -> 2, "big" -> 3
"safety": "low" -> 1, "med" -> 2, "high" -> 3
```
## 4. Results on UCI Breast Cancer Wisconsin (Diagnostic) Dataset

```

```
## 5. Results on UCI Car Evaluation Dataset

```

```
## 6. References
1. [Sebastian Mantey's repository]()
2. [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
3. [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
