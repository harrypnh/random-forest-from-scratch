import numpy
import pandas
import random
from decisionTree import buildDecisionTree, decisionTreePredictions

def loadData(xTrainFile, xTestFile, yTrainFile, yTestFile):
    dataFrameTrain = pandas.read_csv(xTrainFile)
    dataFrameTest = pandas.read_csv(xTestFile)
    dataLabelTrain = pandas.read_csv(yTrainFile)
    dataLabelTest = pandas.read_csv(yTestFile)
    dataFrameTrain = pandas.concat([dataFrameTrain, dataLabelTrain], axis = 1)
    dataFrameTest = pandas.concat([dataFrameTest, dataLabelTest], axis = 1)
    dataFrameTrain = dataFrameTrain.rename(columns = {dataFrameTrain.columns[-1]: "category"})
    dataFrameTest = dataFrameTest.rename(columns = {dataFrameTest.columns[-1]: "category"})
    return dataFrameTrain, dataFrameTest

def bootstrapSample(dataFrame, bootstrapSize):
    randomIndices = numpy.random.randint(low = 0, high = len(dataFrame), size = bootstrapSize)
    return dataFrame.iloc[randomIndices]

def createRandomForest(dataFrame, bootstrapSize, randomAttributes,
                       randomSplits, forestSize = 20, treeMaxDepth = 1000):
    forest = []
    for i in range(forestSize):
        bootstrappedDataFrame = bootstrapSample(dataFrame, bootstrapSize)
        decisionTree = buildDecisionTree(bootstrappedDataFrame, maxDepth = treeMaxDepth,
                                         randomAttributes = randomAttributes,
                                         randomSplits = randomSplits)
        forest.append(decisionTree)
    return forest

def randomForestPredictions(dataFrame, randomForest):
    predictions = {}
    for i in range(len(randomForest)):
        column = "decision tree " + str(i)
        predictions[column] = decisionTreePredictions(dataFrame, randomForest[i])
    predictions = pandas.DataFrame(predictions)
    return predictions.mode(axis = 1)[0]

def calculateAccuracy(predictedResults, category):
    resultCorrect = predictedResults == category
    return resultCorrect.mean()
