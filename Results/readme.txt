Readme


The complete source code and files needed for this machine learning project is contained in source-code directory. The python code for this is 'fullcode.py'

1) The python code trains an SVM model with training dataset and calculates the accuracy and F1 Score after predicting for the test dataset.
2) The python code also loads the already created classifiers from pickle files and tests it with the test data and outputs the accuracy & F1 Score.



***********************************************
		How to Run the Code
***********************************************

1)To Train a model

If the code is run without specifying the file name of the classifiers as shown below, then it trains a model.

Example:

python fullcode.py


2)To use an already trained model (consumes less time!):

If the code is run by specifying the file name as an argument then, it uses the trained model and calculates the accuracy of the model.

Example:

python fullcode.py classifier1.pkl

Note: be sure to enter the filename correctly.

The various available classifiers are given below


***********************************************
		Classifiers
***********************************************

The following classifiers have already been trained with 'training dataset'
The files and the corresponding C and gamma values are specified.
classifier1.pkl    C=100,gamma=10
classifier2.pkl    C=100,gamma=1
classifier3.pkl    C=100,gamma=0.0243 (auto)
classifier4.pkl    C=1,gamma=0.0243 (auto)


Label Encoders:

The label encoders used are saved using pickle modules. The details are given below.

***********************************************
		Label Encoder
***********************************************
The label encoders for 4 fields, are stored in the files below
encoder1.pkl   2nd field of dataset   protocol_type 
encoder2.pkl   3rd field of dataset   service
encoder3.pkl   4th field of dataset   flag
encoder4.pkl   42st field of dataset  attack_type





***********************************************
		Library versions
***********************************************

Python 2.7.11 with Anaconda 2.3.0(64 bit)
Scikit Learn 0.14.1
Numpy 1.9.2
matplotlib 1.4.3
pandas 0.16.2
