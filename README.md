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
Random Forest - Car Evaluation
  Maximum bootstrap size (n) is 1209
  Maximum random subspace size (d) is 6

  Change n, keep other parameters
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 93.06%, accTrain = 95.29%, buildTime = 0.46s
  n = 300, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 91.52%, accTrain = 94.04%, buildTime = 0.47s
  n = 400, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 89.60%, accTrain = 93.47%, buildTime = 0.64s
  n = 500, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 93.83%, accTrain = 95.78%, buildTime = 0.66s
  n = 600, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 93.83%, accTrain = 96.69%, buildTime = 0.74s
  n = 700, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 93.06%, accTrain = 97.19%, buildTime = 0.79s
  n = 800, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.17%, accTrain = 94.79%, buildTime = 0.83s
  n = 900, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 86.32%, accTrain = 90.57%, buildTime = 0.89s
  n = 1000, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 93.06%, accTrain = 96.44%, buildTime = 0.97s
  n = 1100, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 92.87%, accTrain = 97.35%, buildTime = 0.99s
  n = 1200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 92.29%, accTrain = 97.68%, buildTime = 1.16s

  Change d, keep other parameters
  n = 200, d = 1, s = 10, k = 10, maxDepth = 8:
    accTest = 68.98%, accTrain = 70.47%, buildTime = 0.08s
  n = 200, d = 2, s = 10, k = 10, maxDepth = 8:
    accTest = 68.98%, accTrain = 70.72%, buildTime = 0.20s
  n = 200, d = 3, s = 10, k = 10, maxDepth = 8:
    accTest = 80.54%, accTrain = 81.97%, buildTime = 0.67s
  n = 200, d = 4, s = 10, k = 10, maxDepth = 8:
    accTest = 83.43%, accTrain = 87.26%, buildTime = 0.63s
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 88.44%, accTrain = 91.32%, buildTime = 0.49s
  n = 200, d = 6, s = 10, k = 10, maxDepth = 8:
    accTest = 93.64%, accTrain = 95.37%, buildTime = 0.37s

  Change s, keep other parameters
  n = 200, d = 5, s = 2, k = 10, maxDepth = 8:
    accTest = 86.51%, accTrain = 88.42%, buildTime = 0.26s
  n = 200, d = 5, s = 4, k = 10, maxDepth = 8:
    accTest = 89.79%, accTrain = 92.80%, buildTime = 0.29s
  n = 200, d = 5, s = 6, k = 10, maxDepth = 8:
    accTest = 89.40%, accTrain = 92.80%, buildTime = 0.37s
  n = 200, d = 5, s = 8, k = 10, maxDepth = 8:
    accTest = 91.52%, accTrain = 94.46%, buildTime = 0.38s
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.94%, accTrain = 93.71%, buildTime = 0.51s
  n = 200, d = 5, s = 12, k = 10, maxDepth = 8:
    accTest = 90.94%, accTrain = 93.96%, buildTime = 0.60s
  n = 200, d = 5, s = 14, k = 10, maxDepth = 8:
    accTest = 90.94%, accTrain = 94.13%, buildTime = 0.54s
  n = 200, d = 5, s = 16, k = 10, maxDepth = 8:
    accTest = 88.63%, accTrain = 92.89%, buildTime = 0.61s
  n = 200, d = 5, s = 18, k = 10, maxDepth = 8:
    accTest = 90.94%, accTrain = 93.88%, buildTime = 0.87s
  n = 200, d = 5, s = 20, k = 10, maxDepth = 8:
    accTest = 88.25%, accTrain = 91.98%, buildTime = 0.81s

  Change k, keep other parameters
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.17%, accTrain = 94.13%, buildTime = 0.42s
  n = 200, d = 5, s = 10, k = 20, maxDepth = 8:
    accTest = 90.37%, accTrain = 95.45%, buildTime = 1.08s
  n = 200, d = 5, s = 10, k = 30, maxDepth = 8:
    accTest = 92.29%, accTrain = 96.28%, buildTime = 1.42s
  n = 200, d = 5, s = 10, k = 40, maxDepth = 8:
    accTest = 93.64%, accTrain = 96.03%, buildTime = 1.88s
  n = 200, d = 5, s = 10, k = 50, maxDepth = 8:
    accTest = 93.83%, accTrain = 96.53%, buildTime = 2.36s
  n = 200, d = 5, s = 10, k = 60, maxDepth = 8:
    accTest = 91.14%, accTrain = 96.36%, buildTime = 2.84s
  n = 200, d = 5, s = 10, k = 70, maxDepth = 8:
    accTest = 92.87%, accTrain = 96.61%, buildTime = 3.25s
  n = 200, d = 5, s = 10, k = 80, maxDepth = 8:
    accTest = 93.45%, accTrain = 97.85%, buildTime = 3.77s
  n = 200, d = 5, s = 10, k = 90, maxDepth = 8:
    accTest = 92.87%, accTrain = 95.86%, buildTime = 4.31s
  n = 200, d = 5, s = 10, k = 100, maxDepth = 8:
    accTest = 92.68%, accTrain = 97.27%, buildTime = 4.90s
```
## 6. References
1. [Sebastian Mantey's repository](https://github.com/SebastianMantey/Random-Forest-from-Scratch)
2. [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
3. [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
