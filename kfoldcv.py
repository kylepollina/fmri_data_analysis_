from sklearn import svm
import numpy as np
k = 5

# returns a vector of 5 mean squared errors
def svm5F(X, y):
	#line to change for the other model
	clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
	n, d = X.shape
	z = np.zeroes(k)
	for i in range(k):
		#get indices of the current fold
		startIndex = int(n * i / k)
		endIndex = int(n * (i + 1) / k)
		T = set(range(startIndex, endIndex))
		S = set(range(n)) - T #set difference

		#fit the model
		#line to change for the other model
		clf.fit(X[list(S)], y[list(S)])

		#for every incorrectly predicted point, increase the error for that point by 1
		z[i] = 0
		for t in T:
			if y[t] != clf.predict(X[t]):
				z[i] += 1
		#then divide by the number of test points for a weighted error
		z[t] /= len(T)

	return z
