from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import datasets, linear_model,svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import gd
DIC = dict() # genre -> i
ID_GEN = dict()# id -> genre
def GetArtistData():
    df = pd.read_csv('data_by_artist.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12",
                  "Col13", "Col14", "Col15", "Col16"]

    X = df[["Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13"]]
    X = np.array(X)
    ArtistId = df[["Col2"]]
    ArtistId = np.array(ArtistId)[:,0]
    r_X = [] 
    r_id = []
    
    for i in range(len(ArtistId)):
        if ArtistId[i] in ID_GEN:
            r_X.append(X[i])
            r_id.append(ArtistId[i])

    r_X = (r_X - np.min(r_X, axis=0))/(np.max(r_X, axis=0)-np.min(r_X,axis=0))

    return r_X, r_id

def AllMusicionType():
    df = pd.read_csv('influence_data.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

    list1 = df[["Col1", "Col3"]]  # 抽取前八列作为训练数据的各属性值
    list2 = df[["Col5", "Col7"]]   # id genre

    temp1 = np.array(list1)
    temp2 = np.array(list2)

    for _ in temp1:
        ID_GEN[_[0]] = _[1]

    for _ in temp2:
        ID_GEN[_[0]] = _[1]

    return ID_GEN

def CreateMap():
    genre = ["Vocal", "Stage & Screen", "Religious", "Reggae", "R&B;",
             "Pop/Rock", "New Age", "Latin", "Jazz", "International", "Unknown",
             "Folk", "Electronic", "Easy Listening", "Country", "Comedy/Spoken", "Classical",
             "Children's", "Blues", "Avant-Garde"]
    for i in range(len(genre)):
        DIC[genre[i]] = i

# def get_weight(X, y):
#     print("start svm ...")
#     classifier = SVC(kernel='linear', C=1)
#     svm = classifier.fit(X, y)
#     b = classifier.intercept_
#     w = classifier.coef_

#     return w, b

def train(X, y, X_test, y_test):
    # svm_clf = svm.LinearSVC(C = 30, max_iter=3000)
    svm_clf = svm.SVC(C=100, kernel='rbf', gamma=20)
    svm_clf.fit( X, y )
    res = svm_clf.predict(X_test)

    cnt = 0
    for i in range(len(X_test)):
        if res[i] == y_test[i]:
            cnt += 1
    print(cnt/len(X_test))


def show_matrix(M):
    for i in M:
        for j in i:
            print('%4.1f'%j, end=" ")
        print("")
def show_vector(v):
    for i in v:
        print('%4.1f'%i, end=" ")
    print("")

def select_dim(X, l):
    r = [[]  for _ in range(len(X))]
    for j in l:
        for i in range(len(X)):
            r[i].append(X[i][j])

    return r

def getl(X, k):
    l = []
    if k == len(X):
        return X
    for i in range(len(X[0])):
        if k != i:
            l.append(i)
    return l


if __name__ == "__main__":
    CreateMap()
    ID_GEN = AllMusicionType()
    X, y = gd.GetAllData()
    X = X[0:30000]
    y = y[0:30000]
    l = [_ for _ in range(7)]
    print(l)
    tmpX = select_dim(X, l)
    # X, r_id = GetArtistData() # X 0 ~ 10
    pca = PCA(n_components=7)
    pca.fit(X)
    show_vector(pca.explained_variance_ratio_)
    show_vector(pca.explained_variance_)
    
    tmpX = select_dim(X, l)
    X_train, X_test, y_train, y_test = train_test_split(tmpX, y, test_size=.3)
    train(X_train, y_train, X_train, y_train)

    l = [_ for _ in range(5)]
    tmpX = select_dim(X, l)
    X_train, X_test, y_train, y_test = train_test_split(tmpX, y, test_size=.3)
    train(X_train, y_train, X_train, y_train)
    # l = [0,1,2,3,4]
    # print(l)
    # tmpX = select_dim(X, l)
    # X_train, X_test, y_train, y_test = train_test_split(tmpX, y, test_size=.2)
    # train(X_train, y_train, X_train, y_train)

    # print(y)
    # y = []
    # for _ in r_id:
    #     y.append(DIC[ID_GEN[_]])
    

    # # # train(tmpX, y, tmpX, y)

    # for i in range(len(X[0]) + 1):
    #     l = getl(X, i)
    #     print(l)
    #     tmpX = select_dim(X, l)
    #     X_train, X_test, y_train, y_test = train_test_split(tmpX, y, test_size=.2)
    #     train(X_train, y_train, X_test, y_test)
    
    
    # # w, b = get_weight(X, y)


