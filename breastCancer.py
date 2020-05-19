import random
import time
import pandas
from randomForest import trainTestSplit, createRandomForest, randomForestPredictions, calculateAccuracy

dataFrame = pandas.read_csv("dataset_files/breast_cancer.csv")
dataFrame = dataFrame.drop("id", axis = 1)
dataFrame = dataFrame[dataFrame.columns.tolist()[1: ] + dataFrame.columns.tolist()[0: 1]]
dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize = 0.25)

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
