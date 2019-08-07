import numpy as np
import Code_AdaBoost
import Code_StumpClassify


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('horseColicTraining.txt')
    weakClassArr, aggClassEst = Code_AdaBoost.adaBoostTrainDS(dataArr, LabelArr, numIt=40)
    testArr, testLabelArr = loadDataSet('horseColicTest.txt')
    print(weakClassArr)
    predictions = Code_AdaBoost.adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = Code_AdaBoost.adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
