"""ordinary least squares 常规最小二乘法
"""

import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def plotDataSet():
    """绘制数据集
    :return:
    """
    xArr, yArr = loadDataSet('ex0.txt')
    n = len(xArr)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def plotRegression():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    ax.plot(xCopy[:, 1], yHat, c='red')
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()



if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat))
