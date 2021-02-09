from sklearn.svm import SVC
import numpy as np
import pandas as pd
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

def get_weight(X, y):
    print("start svm ...")
    classifier = SVC(kernel='linear', C=1)
    svm = classifier.fit(X, y)
    b = classifier.intercept_
    w = classifier.coef_

    return w, b

def show_matrix(M):
    for i in M:
        for j in i:
            print('%4.1f'%j, end=" ")
        print("")


if __name__ == "__main__":
    CreateMap()
    ID_GEN = AllMusicionType()
    X, r_id = GetArtistData()
    
    y = []
    for _ in r_id:
        y.append(DIC[ID_GEN[_]])
    
    w, b = get_weight(X, y)
    print(len(w))
    print(len(w[0]))
    print(" ")
    show_matrix(w)

    sum = [0] * len(w[0])

    for j in range(len(w[0])):
        tmp = 0
        for i in range(len(w)):
            tmp += w[i][j]
        sum[j] = tmp
    
    for _ in sum:
        print("%.2f"%_, end=" ")
    print("")
