import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIC = dict()

def GetArtistData():
    df = pd.read_csv('data_by_artist.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12",
                  "Col13", "Col14", "Col15", "Col16"]

    X = df[["Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9"]]
    X = np.array(X)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    ArtistId = df[["Col2"]]
    ArtistId = np.array(ArtistId)
    return X,ArtistId

def CreateMap():
    genre = ["Vocal", "Stage & Screen", "Religious", "Reggae", "R&B;",
             "Pop/Rock", "New Age", "Latin", "Jazz", "International", "Unknown",
             "Folk", "Electronic", "Easy Listening", "Country", "Comedy/Spoken", "Classical",
             "Children's", "Blues", "Avant-Garde"]
    for i in range(len(genre)):
        DIC[genre[i]] = i

def GetDistance(ArtistDataArr):
    numlist = [[0]*(len(DIC)) for i in range((len(DIC)))]
    Distance = [[0]*(len(ArtistDataArr)) for i in range((len(ArtistDataArr)))]
    GenreDis = [[0]*(len(DIC)) for i in range((len(DIC)))]
    for i in range(len(ArtistDataArr)):
        for j in range(len(ArtistDataArr)):
            if MusicionType[i] in DIC.keys() and MusicionType[j] in DIC.keys():
                numlist[DIC[MusicionType[i]]][DIC[MusicionType[j]]]+= 1
                Distance[i][j] = np.sqrt(np.sum(np.square(ArtistDataArr[i]-ArtistDataArr[j])))
                GenreDis[DIC[MusicionType[i]]][DIC[MusicionType[j]]] += Distance[i][j]
    GenreDis = np.array(GenreDis)
    numlist = np.array(numlist)
    GenreDis = GenreDis/numlist
    return Distance,GenreDis

def AllMusicionType():
    df = pd.read_csv('influence_data.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

    list1 = df[["Col1", "Col3"]]  # 抽取前八列作为训练数据的各属性值
    list2 = df[["Col5", "Col7"]]

    temp1 = np.array(list1)
    temp2 = np.array(list2)
    temp = np.concatenate((temp1,temp2),axis=0)

    MusicionType = np.array(list(set([tuple(t) for t in temp])))
    return MusicionType

def GetMusicionType(UnorderType,ArtistId,section):
    MusicionType = [None for i in range(len(ArtistId))]
    for i in range(len(ArtistId)):
        tempId = str(ArtistId[i][0])
        if tempId in section:
            ind = section.index(tempId)
            MusicionType[i] = UnorderType[ind][1]
    return MusicionType

global ArtistDataArr
global DistanceArr
global MusicionType
global GenreDistance
if __name__ == "__main__":
    CreateMap()
    ArtistDataArr,ArtistId = GetArtistData()
    all_musicion_type = AllMusicionType().tolist()
    section = [i[0] for i in all_musicion_type]
    MusicionType = GetMusicionType(all_musicion_type, ArtistId, section)

    DistanceArr,GenreDistance = GetDistance(ArtistDataArr)
    PercentArr = [0]*len(GenreDistance)
    for i in range(len(GenreDistance)):
        for j in range(len(GenreDistance)):
            if(GenreDistance[i][j]>GenreDistance[i][i]):
                PercentArr[i] += 1
    PercentArr = np.array(PercentArr)
    PercentArr = PercentArr / (len(GenreDistance))
    print(PercentArr)
