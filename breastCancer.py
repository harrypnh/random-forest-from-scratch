import random
import time
from randomForest import loadData, createRandomForest, randomForestPredictions, calculateAccuracy

xTrainFile = "dataset_files/cancer_X_train.csv"
xTestFile = "dataset_files/cancer_X_test.csv"
yTrainFile = "dataset_files/cancer_y_train.csv"
yTestFile = "dataset_files/cancer_y_test.csv"
dataFrameTrain, dataFrameTest = loadData(xTrainFile, xTestFile, yTrainFile, yTestFile)

print("Random Forest - Breast Cancer Dataset")
print("  Maximum bootstrap size (n) is {}".format(dataFrameTrain.shape[0]))
print("  Maximum random subspace size (d) is {}".format(dataFrameTrain.shape[1] - 1))

print("\n  Change n, keep other parameters")
for i in range(10, dataFrameTrain.shape[0] + 1, 50):
    startTime = time.time()
    randomForest = createRandomForest(dataFrameTrain, bootstrapSize = i,
                                                    randomAttributes = 10, randomSplits = 50,
                                                    forestSize = 30, treeMaxDepth = 3)
    buildingTime = time.time() - startTime
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(i, 10, 50, 30, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")

print("\n  Change d, keep other parameters")
for i in range(10, dataFrameTrain.shape[1], 2):
    startTime = time.time()
    randomForest = createRandomForest(dataFrameTrain, bootstrapSize = 60,
                                                    randomAttributes = i, randomSplits = 50,
                                                    forestSize = 30, treeMaxDepth = 3)
    buildingTime = time.time() - startTime
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(60, i, 50, 30, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")

print("\n  Change s, keep other parameters")
for i in range(10, 100 + 1, 10):
    startTime = time.time()
    randomForest = createRandomForest(dataFrameTrain, bootstrapSize = 60,
                                                    randomAttributes = 10, randomSplits = i,
                                                    forestSize = 30, treeMaxDepth = 3)
    buildingTime = time.time() - startTime
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(60, 10, i, 30, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")

print("\n  Change k, keep other parameters")
for i in range(10, 100 + 1, 10):
    startTime = time.time()
    randomForest = createRandomForest(dataFrameTrain, bootstrapSize = 60,
                                                    randomAttributes = 10, randomSplits = 50,
                                                    forestSize = i, treeMaxDepth = 3)
    buildingTime = time.time() - startTime
    randomForestTestResults = randomForestPredictions(dataFrameTest, randomForest)
    accuracyTest = calculateAccuracy(randomForestTestResults, dataFrameTest.iloc[:, -1]) * 100
    randomForestTrainResults = randomForestPredictions(dataFrameTrain, randomForest)
    accuracyTrain = calculateAccuracy(randomForestTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("  n = {}, d = {}, s = {}, k = {}, maxDepth = {}:".format(60, 10, 50, i, 3))
    print("    accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
