from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn


def train_svm(X_train, y_train):
    svm = SVC()
    svm.fit(X_train, y_train)
    return svm