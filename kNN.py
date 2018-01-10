from numpy import *
import operator

#创建数据集
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 	#4 examples
	labels = ['A','A','B','B'] 		#two classes
	return group, labels

#kNN
#inputX, dataset, lables, no.closest-points
def classify0(intX, dataSet, labels, k):
	#计算距离
	dataSetSize = dataSet.shape[0]		#no. rows of matrix(no.data
	diffMat = tile(intX, (dataSetSize, 1)) - dataSet 	#输入值与每个点的差
	sqDiffMat = diffMat**2 		#差的平方
	sqDistances = sqDiffMat.sum(axis = 1) 	#平方相加
	distances = sqDistances**0.5	#开根号得距离
	#选择距离最小的k个点
	sortedDistIndicies = distances.argsort() 	#返回index
	classCount = {}
	for i in range(k):
		voteLabel = labels[sortedDistIndicies[i]] #距离由近到远点的类
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 #计数{'A':n, 'B':m}
	#比较k个点中每类出现的次数
	maxCount = 0  
	for key, value in classCount.items():
		if value > maxCount:
			maxCount = value 	#类的次数
			maxIndex = key 		#类
	#返回类名
	return maxIndex

'''
>>> import kNN
>>> group,labels = kNN.createDataSet()
>>> group
array([[ 1. ,  1.1],
       [ 1. ,  1. ],
       [ 0. ,  0. ],
       [ 0. ,  0.1]])
>>> labels
['A', 'A', 'B', 'B']
>>> kNN.classify0([0,0],group,labels,3)
'B'
'''
