'''
numpy实现k近邻法（标准k近邻）
数据集：
(1.0,1.1)   A
(1.0,1.0)   A
(0,0)       B
(0,0.1)     B
实验数据：(0.7,0.7)
k = 1
'''

from numpy import *
import operator

def createDataSet():
    # 创建数据集及标签，实际应用可将此部分替换为对目标数据的处理
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):
    # 分类器
    # dataSetSize 为数据集大小
    dataSetSize = dataSet.shape[0]
    # diffMat 为目标数据和数据集中点的坐标差
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    # 计算欧式距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    # 将距离排序
    sortedDistIndicies = distances.argsort()
    # 遍历取出前k个点，利用其类别投票
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

group,labels = createDataSet()
print("DataSet:\n-------\nGroup:")
print(group)
print("------\nLabels:")
print(labels)
inA = [0.7,0.7]
print("------\nResult:")
labelOfA = classify(inA,group,labels,1)
print(labelOfA)