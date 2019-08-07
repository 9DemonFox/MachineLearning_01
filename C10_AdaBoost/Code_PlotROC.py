"""绘制ROC曲线
"""
import Code_AdaBoost
import Code_AdaBoostHorse
import numpy as np
import matplotlib.pyplot as plt
import Code_StumpClassify


def plotROC(predStrengths, classLabels):
    """绘制ROC曲线
    :param preStrengths: 分类器预测强度
    :param classLabels: 类别
    :return:
    """
    font = "SimHei"
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = np.sum(np.array(classLabels) == 1.0)  # 计算正类的数目
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndices = predStrengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndices.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    print('AUC面积为:', ySum * xStep)  # 计算AUC
    plt.show()


if __name__ == '__main__':
    dataArr, LabelArr = Code_AdaBoostHorse.loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = Code_AdaBoost.adaBoostTrainDS(dataArr, LabelArr, numIt=20)
    plotROC(aggClassEst.T, LabelArr)
