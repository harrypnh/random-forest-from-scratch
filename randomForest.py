import random
import time
from decisionTree import *

def calculateAccuracy(predictedResults, category):
    resultCorrect = predictedResults == category
    return resultCorrect.mean()

def bootstrapSample(dataFrame, bootstrapSize):
    randomIndices = numpy.random.randint(low = 0, high = len(dataFrame), size = bootstrapSize)
    return dataFrame.iloc[randomIndices]

def createRandomForest(dataFrame, bootstrapSize, randomAttributes,
                       randomSplits, forestSize = 20, treeMaxDepth = 1000):
    startTime = time.time()
    forest = []
    for i in range(forestSize):
        bootstrappedDataFrame = bootstrapSample(dataFrame, bootstrapSize)
        decisionTree = buildDecisionTree(bootstrappedDataFrame, maxDepth = treeMaxDepth,
                                         randomAttributes = randomAttributes,
                                         randomSplits = randomSplits)
        forest.append(decisionTree)
    return forest, time.time() - startTime

def randomForestPredictions(dataFrame, randomForest):
    predictions = {}
    for i in range(len(randomForest)):
        column = "decision tree " + str(i)
        predictions[column] = decisionTreePredictions(dataFrame, randomForest[i])
    predictions = pandas.DataFrame(predictions)
    return predictions.mode(axis = 1)[0]