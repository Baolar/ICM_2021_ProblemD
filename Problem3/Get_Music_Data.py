import numpy as np
import pandas as pd

genre = ["Vocal", "Stage & Screen", "Religious", "Reggae", "R&B;",
         "Pop/Rock", "New Age", "Latin", "Jazz", "International", "Unknown",
         "Folk", "Electronic", "Easy Listening", "Country", "Comedy/Spoken", "Classical",
         "Children's", "Blues", "Avant-Garde"]
GENRE_DIC = dict()

def GetMusicData():
    df = pd.read_csv('2021_ICM_Problem_D_Data\\full_music_data.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8","Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16","Col17", "Col18", "Col19"]

    X = df[["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8","Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16","Col17", "Col18", "Col19"]]  # 抽取前八列作为训练数据的各属性值
    X = np.array(X)
    ArtistId = df[["Col2"]]
    ArtistId = np.array(ArtistId)
    return X, ArtistId

def Unification(arr):
    temp = arr.T[np.arange(2,13)]
    data = temp.T  # 提取后再将转置为列，*重要*
    data = (data - np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))
    return data

def AllMusicionType():
    df = pd.read_csv('2021_ICM_Problem_D_Data\influence_data.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

    list1 = df[["Col1", "Col3"]]  # 抽取前八列作为训练数据的各属性值
    list2 = df[["Col5", "Col7"]]

    temp1 = np.array(list1)
    temp2 = np.array(list2)
    temp = np.concatenate((temp1,temp2),axis=0)

    MusicionType = np.array(list(set([tuple(t) for t in temp])))
    return MusicionType

def GetMusicionType(UnorderType,ArtistId,section,MusicData):
    MusicionType = []
    MusicArr = []
    for i in range(len(ArtistId)):
        tempId = str(ArtistId[i][0])
        tempId = tempId.lstrip('[')
        tempId = tempId.rstrip(']')
        if tempId in section:
            ind = section.index(tempId)
            MusicionType.append(GENRE_DIC[UnorderType[ind][1]])
            MusicArr.append(MusicData[i])
    return MusicionType,MusicArr

global MusicData,ArtistId

def GetAllData():
    for i in range(len(genre)):
        GENRE_DIC[genre[i]] = i
    MusicData,ArtistId = GetMusicData()
    MusicData = Unification(MusicData)
    all_musicion_type = AllMusicionType().tolist()
    section = [i[0] for i in all_musicion_type]
    MusicionType,MusicArr = GetMusicionType(all_musicion_type, ArtistId, section,MusicData)
    return MusicArr,MusicionType

if __name__ == "__main__":
    MusicArr,MusicionType = GetAllData()
