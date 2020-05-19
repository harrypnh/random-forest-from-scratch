# Random Forest From Scratch
Random Forest Algorithm written in Python using NumPy and Pandas. Based on the [Decision Tree project](https://github.com/huannpham/decision-tree-from-scratch).
## 1. Overview of the Implemention

## 2. Repository Structure
```
decision-tree-from-scratch/
├── dataset_files/
│   ├── breast_cancer.csv   # UCI Breast Cancer Wisconsin Diagnostic Dataset
│   └── car_evaluation.csv  # UCI Car Evaluation Dataset
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
  Maximum bootstrap size (n) is 427
  Maximum random subspace size (d) is 30

  Change n, keep other parameters
  n = 10, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 92.25%, accTrain = 95.08%, buildTime = 0.23s
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 92.96%, accTrain = 96.02%, buildTime = 0.69s
  n = 110, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 92.96%, accTrain = 97.66%, buildTime = 0.89s
  n = 160, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.07%, accTrain = 94.85%, buildTime = 1.11s
  n = 210, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 97.19%, buildTime = 1.27s
  n = 260, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 94.37%, accTrain = 97.42%, buildTime = 1.42s
  n = 310, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 94.37%, accTrain = 98.59%, buildTime = 1.59s
  n = 360, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 96.96%, buildTime = 1.73s
  n = 410, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 96.48%, accTrain = 97.42%, buildTime = 1.85s

  Change d, keep other parameters
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 96.72%, buildTime = 0.75s
  n = 60, d = 12, s = 50, k = 30, maxDepth = 3:
    accTest = 97.18%, accTrain = 96.96%, buildTime = 0.69s
  n = 60, d = 14, s = 50, k = 30, maxDepth = 3:
    accTest = 96.48%, accTrain = 95.08%, buildTime = 0.65s
  n = 60, d = 16, s = 50, k = 30, maxDepth = 3:
    accTest = 93.66%, accTrain = 95.78%, buildTime = 0.67s
  n = 60, d = 18, s = 50, k = 30, maxDepth = 3:
    accTest = 92.96%, accTrain = 95.78%, buildTime = 0.63s
  n = 60, d = 20, s = 50, k = 30, maxDepth = 3:
    accTest = 93.66%, accTrain = 96.49%, buildTime = 0.67s
  n = 60, d = 22, s = 50, k = 30, maxDepth = 3:
    accTest = 97.18%, accTrain = 96.25%, buildTime = 0.73s
  n = 60, d = 24, s = 50, k = 30, maxDepth = 3:
    accTest = 93.66%, accTrain = 96.02%, buildTime = 0.63s
  n = 60, d = 26, s = 50, k = 30, maxDepth = 3:
    accTest = 92.25%, accTrain = 96.25%, buildTime = 0.73s
  n = 60, d = 28, s = 50, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 96.25%, buildTime = 0.69s
  n = 60, d = 30, s = 50, k = 30, maxDepth = 3:
    accTest = 92.96%, accTrain = 96.02%, buildTime = 0.70s

  Change s, keep other parameters
  n = 60, d = 10, s = 10, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 94.85%, buildTime = 0.19s
  n = 60, d = 10, s = 20, k = 30, maxDepth = 3:
    accTest = 92.96%, accTrain = 96.02%, buildTime = 0.32s
  n = 60, d = 10, s = 30, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 96.25%, buildTime = 0.44s
  n = 60, d = 10, s = 40, k = 30, maxDepth = 3:
    accTest = 91.55%, accTrain = 94.38%, buildTime = 0.51s
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.77%, accTrain = 96.02%, buildTime = 0.64s
  n = 60, d = 10, s = 60, k = 30, maxDepth = 3:
    accTest = 95.07%, accTrain = 95.08%, buildTime = 0.70s
  n = 60, d = 10, s = 70, k = 30, maxDepth = 3:
    accTest = 93.66%, accTrain = 94.61%, buildTime = 0.87s
  n = 60, d = 10, s = 80, k = 30, maxDepth = 3:
    accTest = 94.37%, accTrain = 96.49%, buildTime = 1.01s
  n = 60, d = 10, s = 90, k = 30, maxDepth = 3:
    accTest = 93.66%, accTrain = 94.85%, buildTime = 1.14s
  n = 60, d = 10, s = 100, k = 30, maxDepth = 3:
    accTest = 96.48%, accTrain = 95.78%, buildTime = 1.17s

  Change k, keep other parameters
  n = 60, d = 10, s = 50, k = 10, maxDepth = 3:
    accTest = 95.07%, accTrain = 94.38%, buildTime = 0.23s
  n = 60, d = 10, s = 50, k = 20, maxDepth = 3:
    accTest = 93.66%, accTrain = 95.55%, buildTime = 0.42s
  n = 60, d = 10, s = 50, k = 30, maxDepth = 3:
    accTest = 95.07%, accTrain = 96.96%, buildTime = 0.65s
  n = 60, d = 10, s = 50, k = 40, maxDepth = 3:
    accTest = 94.37%, accTrain = 95.78%, buildTime = 0.88s
  n = 60, d = 10, s = 50, k = 50, maxDepth = 3:
    accTest = 94.37%, accTrain = 97.19%, buildTime = 1.10s
  n = 60, d = 10, s = 50, k = 60, maxDepth = 3:
    accTest = 92.96%, accTrain = 96.02%, buildTime = 1.38s
  n = 60, d = 10, s = 50, k = 70, maxDepth = 3:
    accTest = 94.37%, accTrain = 97.19%, buildTime = 1.52s
  n = 60, d = 10, s = 50, k = 80, maxDepth = 3:
    accTest = 93.66%, accTrain = 95.78%, buildTime = 1.73s
  n = 60, d = 10, s = 50, k = 90, maxDepth = 3:
    accTest = 94.37%, accTrain = 96.25%, buildTime = 2.03s
  n = 60, d = 10, s = 50, k = 100, maxDepth = 3:
    accTest = 94.37%, accTrain = 96.02%, buildTime = 2.15s
```
## 5. Results on UCI Car Evaluation Dataset

```
Random Forest - Car Evaluation Dataset
  Maximum bootstrap size (n) is 1210
  Maximum random subspace size (d) is 6

  Change n, keep other parameters
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.35%, accTrain = 91.32%, buildTime = 0.46s
  n = 300, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 92.28%, accTrain = 93.72%, buildTime = 0.52s
  n = 400, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.15%, accTrain = 92.48%, buildTime = 0.83s
  n = 500, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 92.66%, accTrain = 95.45%, buildTime = 0.68s
  n = 600, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.73%, accTrain = 95.95%, buildTime = 0.77s
  n = 700, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 92.28%, accTrain = 96.45%, buildTime = 0.83s
  n = 800, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 93.44%, accTrain = 96.03%, buildTime = 0.74s
  n = 900, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 92.66%, accTrain = 95.37%, buildTime = 0.88s
  n = 1000, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.15%, accTrain = 95.37%, buildTime = 1.01s
  n = 1100, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 86.10%, accTrain = 91.90%, buildTime = 1.11s
  n = 1200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 94.79%, accTrain = 96.94%, buildTime = 0.97s

  Change d, keep other parameters
  n = 200, d = 1, s = 10, k = 10, maxDepth = 8:
    accTest = 72.01%, accTrain = 69.17%, buildTime = 0.09s
  n = 200, d = 2, s = 10, k = 10, maxDepth = 8:
    accTest = 72.78%, accTrain = 69.59%, buildTime = 0.21s
  n = 200, d = 3, s = 10, k = 10, maxDepth = 8:
    accTest = 83.20%, accTrain = 81.82%, buildTime = 0.45s
  n = 200, d = 4, s = 10, k = 10, maxDepth = 8:
    accTest = 89.00%, accTrain = 88.26%, buildTime = 0.55s
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 90.93%, accTrain = 93.06%, buildTime = 0.46s
  n = 200, d = 6, s = 10, k = 10, maxDepth = 8:
    accTest = 94.98%, accTrain = 96.28%, buildTime = 0.39s

  Change s, keep other parameters
  n = 200, d = 5, s = 2, k = 10, maxDepth = 8:
    accTest = 84.56%, accTrain = 85.45%, buildTime = 0.19s
  n = 200, d = 5, s = 4, k = 10, maxDepth = 8:
    accTest = 86.49%, accTrain = 88.51%, buildTime = 0.24s
  n = 200, d = 5, s = 6, k = 10, maxDepth = 8:
    accTest = 89.58%, accTrain = 92.15%, buildTime = 0.32s
  n = 200, d = 5, s = 8, k = 10, maxDepth = 8:
    accTest = 89.58%, accTrain = 93.14%, buildTime = 0.39s
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 91.70%, accTrain = 94.71%, buildTime = 0.59s
  n = 200, d = 5, s = 12, k = 10, maxDepth = 8:
    accTest = 93.05%, accTrain = 93.80%, buildTime = 0.68s
  n = 200, d = 5, s = 14, k = 10, maxDepth = 8:
    accTest = 86.49%, accTrain = 90.66%, buildTime = 0.65s
  n = 200, d = 5, s = 16, k = 10, maxDepth = 8:
    accTest = 91.51%, accTrain = 93.88%, buildTime = 0.66s
  n = 200, d = 5, s = 18, k = 10, maxDepth = 8:
    accTest = 90.73%, accTrain = 93.80%, buildTime = 0.75s
  n = 200, d = 5, s = 20, k = 10, maxDepth = 8:
    accTest = 89.77%, accTrain = 92.89%, buildTime = 0.85s

  Change k, keep other parameters
  n = 200, d = 5, s = 10, k = 10, maxDepth = 8:
    accTest = 89.38%, accTrain = 91.82%, buildTime = 0.52s
  n = 200, d = 5, s = 10, k = 20, maxDepth = 8:
    accTest = 90.35%, accTrain = 94.88%, buildTime = 1.00s
  n = 200, d = 5, s = 10, k = 30, maxDepth = 8:
    accTest = 93.24%, accTrain = 96.61%, buildTime = 1.45s
  n = 200, d = 5, s = 10, k = 40, maxDepth = 8:
    accTest = 91.70%, accTrain = 95.45%, buildTime = 2.06s
  n = 200, d = 5, s = 10, k = 50, maxDepth = 8:
    accTest = 93.82%, accTrain = 96.78%, buildTime = 2.34s
  n = 200, d = 5, s = 10, k = 60, maxDepth = 8:
    accTest = 91.89%, accTrain = 95.12%, buildTime = 2.83s
  n = 200, d = 5, s = 10, k = 70, maxDepth = 8:
    accTest = 94.40%, accTrain = 96.12%, buildTime = 3.13s
  n = 200, d = 5, s = 10, k = 80, maxDepth = 8:
    accTest = 92.86%, accTrain = 96.20%, buildTime = 3.64s
  n = 200, d = 5, s = 10, k = 90, maxDepth = 8:
    accTest = 94.79%, accTrain = 97.27%, buildTime = 4.25s
  n = 200, d = 5, s = 10, k = 100, maxDepth = 8:
    accTest = 94.40%, accTrain = 96.78%, buildTime = 4.62s
```
## 6. References
1. [Sebastian Mantey's repository](https://github.com/SebastianMantey/Random-Forest-from-Scratch)
2. [UCI Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
3. [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
