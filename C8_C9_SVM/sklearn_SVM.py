import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC


def img2vector(filename):
    """把图像转为向量
    :param filename:
    :return:
    """
    returnVec = np.zeros((1, 1024))

    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('../C1_KNN/sklearn_KNN/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector('../C1_KNN/sklearn_KNN/trainingDigits/%s' % (fileNameStr))
    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)
    testFileList = listdir('../C1_KNN/sklearn_KNN/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('../C1_KNN/sklearn_KNN/testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = clf.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
