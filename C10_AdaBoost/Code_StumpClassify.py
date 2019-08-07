"""单层分类
"""
import numpy as np
import matplotlib.pyplot as plt


def loadSimpleData():
    """创建单层决策树的数据集
    :return:
    """
    dataMat = np.matrix([[1.0, 2.1],
                         [1.5, 1.6],
                         [1.3, 1.0],
                         [1.0, 1.0],
                         [2.0, 1.0]
                         ])

    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def showDataSet(dataMat, labelMat):
    """数据集可视化
    :return:
    """
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    """单层决策树分类函数
    :param dataMat:
    :param dimen:
    :param threshVal: 阈值
    :param threshIneq: 标志
    :return:
    """
    returnArr = np.ones((np.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        returnArr[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        returnArr[dataMat[:, dimen] > threshVal] = -1.0
    return returnArr


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()  # 找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


if __name__ == '__main__':
    dataArr, classLabels = loadSimpleData()
    showDataSet(dataArr, classLabels)
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
    print('bestStump:\n', bestStump)
    print('minError:\n', minError)
    print('bestClasEst:\n', bestClasEst)
