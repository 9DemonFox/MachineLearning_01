import Code_DataFiting
import GradientAscent
import RandomGradAscent
import numpy as np


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[-1]))
    trainWeights_None_random, _ = RandomGradAscent.stocGradAscent(np.array(trainingSet), trainingLabels)
    trainWeights_random, _ = Code_DataFiting.gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVec = 0
    frTrain.close()
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights_random)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100
    print("随机梯度下降测试集错误率为: %.2f%%" % errorRate)

    frTest.close()
    frTest == open('horseColicTest.txt')

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights_None_random)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100
    print("梯度下降测试集错误率为: %.2f%%" % errorRate)


def classifyVector(inX, weights):
    prob = Code_DataFiting.sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    colicTest()
