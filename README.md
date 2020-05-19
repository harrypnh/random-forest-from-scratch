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
Random Forest - Breast Cancer Dataset
  Maximum bootstrap size (n) is 426
  Maximum random subspace size (d) is 30

  Change n, keep other parameters
  n = 10, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 93.71%, accTrain = 95.31%, buildTime = 0.24s
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 94.41%, accTrain = 96.48%, buildTime = 0.68s
  n = 110, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 94.41%, accTrain = 97.65%, buildTime = 0.77s
  n = 160, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 93.71%, accTrain = 97.18%, buildTime = 0.90s
  n = 210, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 93.71%, accTrain = 97.65%, buildTime = 1.02s
  n = 260, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 96.50%, accTrain = 96.48%, buildTime = 1.09s
  n = 310, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 96.50%, accTrain = 97.65%, buildTime = 1.14s
  n = 360, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 97.90%, accTrain = 97.42%, buildTime = 1.15s
  n = 410, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.48%, buildTime = 1.17s

  Change d, keep other parameters
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 96.50%, accTrain = 96.48%, buildTime = 0.59s
  n = 60, d = 12, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.24%, buildTime = 0.60s
  n = 60, d = 14, s = 50, k = 30, maxDepth = 3:
    accTest = 94.41%, accTrain = 96.24%, buildTime = 0.63s
  n = 60, d = 16, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.71%, buildTime = 0.66s
  n = 60, d = 18, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.24%, buildTime = 0.62s
  n = 60, d = 20, s = 50, k = 30, maxDepth = 3:
    accTest = 94.41%, accTrain = 95.77%, buildTime = 0.65s
  n = 60, d = 22, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 95.77%, buildTime = 0.60s
  n = 60, d = 24, s = 50, k = 30, maxDepth = 3:
    accTest = 93.71%, accTrain = 95.07%, buildTime = 0.65s
  n = 60, d = 26, s = 50, k = 30, maxDepth = 3:
    accTest = 96.50%, accTrain = 95.07%, buildTime = 0.64s
  n = 60, d = 28, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.24%, buildTime = 0.72s
  n = 60, d = 30, s = 50, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.01%, buildTime = 0.66s

  Change s, keep other parameters
  n = 60, d = 10, s = 10, k = 30, maxDepth = 3:
    accTest = 93.01%, accTrain = 94.84%, buildTime = 0.20s
  n = 60, d = 10, s = 20, k = 30, maxDepth = 3:
    accTest = 93.01%, accTrain = 95.77%, buildTime = 0.30s
  n = 60, d = 10, s = 30, k = 30, maxDepth = 3:
    accTest = 93.71%, accTrain = 96.01%, buildTime = 0.40s
  n = 60, d = 10, s = 40, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 97.18%, buildTime = 0.51s
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 94.41%, accTrain = 96.71%, buildTime = 0.58s
  n = 60, d = 10, s = 60, k = 30, maxDepth = 3:
    accTest = 93.71%, accTrain = 96.01%, buildTime = 0.71s
  n = 60, d = 10, s = 70, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 95.54%, buildTime = 0.80s
  n = 60, d = 10, s = 80, k = 30, maxDepth = 3:
    accTest = 92.31%, accTrain = 95.07%, buildTime = 0.93s
  n = 60, d = 10, s = 90, k = 30, maxDepth = 3:
    accTest = 94.41%, accTrain = 96.71%, buildTime = 0.99s
  n = 60, d = 10, s = 100, k = 30, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.48%, buildTime = 1.04s

  Change k, keep other parameters
  n = 60, d = 10, s = 50, k = 10, maxDepth = 3:
    accTest = 95.10%, accTrain = 95.07%, buildTime = 0.19s
  n = 60, d = 10, s = 50, k = 20, maxDepth = 3:
    accTest = 96.50%, accTrain = 96.01%, buildTime = 0.43s
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 96.50%, accTrain = 95.77%, buildTime = 0.57s
  n = 60, d = 10, s = 50, k = 40, maxDepth = 3:
    accTest = 95.80%, accTrain = 95.31%, buildTime = 0.78s
  n = 60, d = 10, s = 50, k = 50, maxDepth = 3:
    accTest = 95.10%, accTrain = 96.71%, buildTime = 1.01s
  n = 60, d = 10, s = 50, k = 60, maxDepth = 3:
    accTest = 95.10%, accTrain = 95.77%, buildTime = 1.16s
  n = 60, d = 10, s = 50, k = 70, maxDepth = 3:
    accTest = 94.41%, accTrain = 96.95%, buildTime = 1.43s
  n = 60, d = 10, s = 50, k = 80, maxDepth = 3:
    accTest = 93.01%, accTrain = 96.71%, buildTime = 1.65s
  n = 60, d = 10, s = 50, k = 90, maxDepth = 3:
    accTest = 96.50%, accTrain = 96.48%, buildTime = 1.81s
  n = 60, d = 10, s = 50, k = 100, maxDepth = 3:
    accTest = 95.80%, accTrain = 96.48%, buildTime = 1.98s
```
## 5. Results on UCI Car Evaluation Dataset

```

```
## 6. References
1. [Sebastian Mantey's repository](https://github.com/SebastianMantey/Random-Forest-from-Scratch)
2. [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
3. [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
