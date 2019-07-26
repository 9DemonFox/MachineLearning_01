from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus


def C3_step1():
    """第三节课第1步
    :return:
    """
    with open('./data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lencesLables = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lenses, lencesLables)


def C3_step2():
    """生成pandas数据 方便序列化工作
    :return:
    """
    with open('./data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])



if __name__ == '__main__':
    C3_step1()
