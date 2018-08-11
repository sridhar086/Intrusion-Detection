import numpy as np
import pandas as pa
import sklearn as sk
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
import pickle
import matplotlib.pyplot as plt
from collections import Counter

def cross_model():
	train_data = pa.read_csv('Train.csv')
	test_data = pa.read_csv('Test.csv')
	
	df = train_data.copy()
	dfa = test_data.copy()
	
	#d=df.ix[:,0:41].corr()
	#plt.matshow(d)
	#plt.colorbar()
	#plt.show()
	
	frames = [df,dfa]
	result = pa.concat(frames)
	y = np.array(result.ix[:,41])
	idx = np.where(y=='normal.')
	idx = list(idx[0])
	attack_idx = list(set(range(len(y))).difference(set(idx)))
	y[attack_idx] = 'attack.'
	Counter(y)
	result.ix[:,41] =y
	#print result.shape
	#print df.shape
	#print dfa.shape

	#Creating the label encoder for the first time
	l1 = preprocessing.LabelEncoder()
	l2 = preprocessing.LabelEncoder()
	l3 = preprocessing.LabelEncoder()
	l4 = preprocessing.LabelEncoder()
	
	l1.fit(result.ix[:,1])
	l2.fit(result.ix[:,2])
	l3.fit(result.ix[:,3])
	l4.fit(result.ix[:,41])

		
	p1 = l1.transform(result.ix[:,1])
	p2 = l2.transform(result.ix[:,2])
	p3 = l3.transform(result.ix[:,3])
	p4 = l4.transform(result.ix[:,41])

	result.ix[:,1] = p1
	result.ix[:,2] = p2
	result.ix[:,3] = p3
	result.ix[:,41] = p4

	dataset = result.as_matrix()
	y_train = dataset[:,41]
	x_t = dataset[:,:-1]

	x_train = preprocessing.normalize(x_t)
	y_train = y_train.astype('int')

	train = np.array(train_data.ix[:,41],dtype='string')
	test = np.array(test_data.ix[:,41],dtype='string')
	# To create a classifier with SVM SVC kernel and specify the parameters C and gamma
	cv = cross_validation.StratifiedKFold(y_train, n_folds=2)
	print "creating a model"
	classifier = svm.SVC(kernel='rbf', C=100,gamma =1)
	print "cross validation"
	
	
	ara = cross_validation.cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc')
	with open('file.pkl','wb') as f:
		pickle.dump(ara,f)

	f.close()

	

if __name__=='__main__':
	cross_model()
	



