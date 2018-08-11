import numpy as np
import pandas as pa
import sklearn as sk
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import sys

global C
global g

def train_model(file):
	train_data = pa.read_csv('Train.csv')
	test_data = pa.read_csv('Test.csv')
	
	df = train_data.copy()
	dfa = test_data.copy()
	
	
	
	frames = [df,dfa]
	result = pa.concat(frames)
	#print result.shape
	#print df.shape
	#print dfa.shape
	

	#Creating the label encoder for the first time
	#l1 = preprocessing.LabelEncoder()
	#l2 = preprocessing.LabelEncoder()
	#l3 = preprocessing.LabelEncoder()
	#l4 = preprocessing.LabelEncoder()

	#l1.fit(result.ix[:,1])
	#l2.fit(result.ix[:,2])
	#l3.fit(result.ix[:,3])
	#l4.fit(df.ix[:,41])


	with open('encoder1.pkl','rb') as f:
		l1 = pickle.load(f)


	with open('encoder2.pkl','rb') as f:
		l2 = pickle.load(f)


	with open('encoder3.pkl','rb') as f:
		l3 = pickle.load(f)

		
	with open('encoder4.pkl','rb') as f:
		l4 = pickle.load(f)
		
	print "Loaded the encoders"	
	p1 = l1.transform(df.ix[:,1])
	p2 = l2.transform(df.ix[:,2])
	p3 = l3.transform(df.ix[:,3])
	p4 = l4.transform(df.ix[:,41])

	df.ix[:,1] = p1
	df.ix[:,2] = p2
	df.ix[:,3] = p3
	df.ix[:,41] = p4
	
	#d=df.ix[:,0:41].corr()
	#plt.matshow(d)
	#plt.colorbar()
	#plt.title('Correlation of the features')
	#plt.show()
	
	#np.linalg.matrix_rank(np.array(df.ix[:,0:41]),tol=50)  This is to find the rank of matrix with a set threshold

	dataset = df.as_matrix()
	y_train = dataset[:,41]
	x_t = dataset[:,:-1]

	x_train = preprocessing.normalize(x_t)
	y_train = y_train.astype('int')

	pa1 = l1.transform(dfa.ix[:,1])
	pa2 = l2.transform(dfa.ix[:,2])
	pa3 = l3.transform(dfa.ix[:,3])

	dfa.ix[:,1] = pa1
	dfa.ix[:,2] = pa2
	dfa.ix[:,3] = pa3

	dataset1 = dfa.as_matrix()
	x_te = dataset1[:,:-1]
	x_te = x_te.astype('float')
	x_test_norm = preprocessing.normalize(x_te)


	train = np.array(train_data.ix[:,41],dtype='string')
	test = np.array(test_data.ix[:,41],dtype='string')
	
	# To create a classifier with SVM SVC kernel and specify the parameters C and gamma
	#classifier = svm.SVC(kernel='rbf',C=100,gamma=1)
	#classifier.fit(x_train,y_train)
	
	if(file=='train'):
		print "Training the model"
		classifier = svm.SVC(kernel='rbf',C=C,gamma=g)
		print "Model created"
		classifier.fit(x_train,y_train)
		print "Model is fit with data"
	else:
		with open(file,'rb') as f:
			classifier = pickle.load(f)
		
		print "Loaded the classifier"
	
	#with open('classifier2.pkl','rb') as f:
	#	classifier = pickle.load(f)
	print classifier
	print "making the prediction"
	prediction = classifier.predict(x_test_norm)
	pred = l4.inverse_transform(prediction)
	pred = pred.astype('string')
	#print pred
	#print Counter(pred)
	#print pred.shape
	
	accuracy(train,test,pred)
	
def accuracy(train,test,pred):
	common = set(train).symmetric_difference(set(test))

	index=[]
	for i in common:
		index.extend(np.where(test==i))


	flat = []
	for lis in index:
		for elem in lis:
			flat.append(elem.astype('int'))

	full = range(len(test))
	elements = list(set(full).difference(set(flat)))
	res = pred[elements] == test[elements]
	
	from sklearn.metrics import accuracy_score
	pred=pred.astype('string')
	test=test.astype('string')
	#print type(pred)
	#print type(test)
	#print pred.dtype
	#print test.dtype
	#print 'the predicted ones',Counter(pred)
	#print 'the test ',Counter(test)
	#print 'The result',Counter(res)
	print 'Accuracy without considering the unknown attack types',accuracy_score(test[elements],pred[elements])
	print 'Accuracy considering the unknown attack types',accuracy_score(test,pred)

	new_pred = pred[elements]
	new_test = test[elements]
	print 'F1 Score without considering the unknown attack types',f1score(new_pred,new_test)
	new_pred = np.copy(pred)
	new_test = np.copy(test)
	print 'F1 Score considering the unknown attack types',f1score(new_pred,new_test)

def f1score(new_pred,new_test):
	
	non_attack = np.where(new_test=='normal.')
	non_attack = list(non_attack[0])
	n_full = range(len(new_test))
	normal_count = len(non_attack)
	attack_count = len(n_full) - normal_count

	attack = list(set(n_full).difference(set(non_attack)))

	fn = len(np.where(new_pred[attack]=='normal.')[0])
	tp = attack_count-fn

	tn = len(np.where(new_pred[non_attack]=='normal.')[0])
	fp = normal_count - tn

	#print tp,fn

	prec = float(tp)/float((tp+fp))
	rec = float(tp)/float((tp+fn))
	f1score = (2*prec*rec)/(prec+rec)
	return f1score


	
if __name__=='__main__':
	global C
	global g
	C=100
	g=1
	if len(sys.argv) == 1:
		print "Training the model with C = ",C," and gamma = ",g
		train_model('train')
		
	else:
		train_model(sys.argv[1])
		
	

	
	



