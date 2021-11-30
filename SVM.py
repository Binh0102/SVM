import numpy as np
from sklearn.svm import SVC

X1 = [[1,3], [3,3], [4,0], [3,0], [2, 2]]
y1 = [1, 1, 1, 1, 1]
X2 = [[0,0], [1,1], [1,2], [2,0]]
y2 = [-1, -1, -1, -1]
X = np.array(X1 + X2)
y = y1 + y2

clf = SVC(kernel='linear', C=1E10)
clf.fit(X, y)
print (clf.support_vectors_)