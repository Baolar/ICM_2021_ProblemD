import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False

genre = ["Vocal", "Stage & Screen", "Religious", "Reggae", "R&B;",
             "Pop/Rock", "New Age", "Latin", "Jazz", "International", "Unknown",
             "Folk", "Electronic", "Easy Listening", "Country", "Comedy/Spoken", "Classical",
             "Children's", "Blues", "Avant-Garde"]

year = ['1920','1930','1940','1950','1960','1970','1980','1990','2000','2010']

GENRE_DIC = dict()
YEAR_DIC = dict()

def GetReadyData():
    for i in range(len(genre)):
        GENRE_DIC[genre[i]] = i
    for i in range(len(year)):
        YEAR_DIC[year[i]] = i

    df = pd.read_csv('2021_ICM_Problem_D_Data\\full_music_data.csv')
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8","Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16","Col17", "Col18", "Col19"]  # 抽取前八列作为训练数据的各属性值
    X = df[["Col2","Col17"]]
    X = np.array(X)
    MusicArr = []
    for i in range(len(X)):
        temp_year = (X[i][1]//10)*10
        if str(temp_year) in YEAR_DIC.keys():
            MusicArr.append([X[i][0],temp_year])
    return MusicArr

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
    return MusicionType, MusicArr

def GetSimpleMatrix(Num_Matrix):
    Minor = []
    SimpleMatrix = []
    SimpleGenre = []
    for i in range(len(Num_Matrix)):
        if max(Num_Matrix[i]) <= 650:
            Minor.append(Num_Matrix[i])
        else:
            SimpleMatrix.append(Num_Matrix[i])
            SimpleGenre.append(genre[i])
    SimpleMatrix.append(np.mean(Minor,axis=0))
    SimpleGenre.append("Minority music")
    SimplePercentArr = SimpleMatrix/(np.sum(SimpleMatrix,axis=0))*100
    return SimplePercentArr,SimpleGenre

def CreateLineChart(SimplePercentArr,SimpleGenre):
    x = year
    for i in range(len(SimplePercentArr)):
        y = SimplePercentArr[i]
        plt.plot(x,y,marker='.')
    plt.legend(SimpleGenre, loc='lower left', bbox_to_anchor=(0.77, 0.2))
    plt.title("The way genres change over time.")
    plt.xlabel("The Year")
    plt.ylabel("The Percentage")
    plt.show()

def ShowChart():
    MusicArr = GetReadyData()

    all_musicion_type = AllMusicionType().tolist()
    section = [i[0] for i in all_musicion_type]
    MusicionType, MusicArr = GetMusicionType(all_musicion_type, MusicArr, section, MusicArr)

    Num_Matrix = [[0] * (len(year)) for i in range((len(genre)))]
    for i in range(len(MusicArr)):
        Num_Matrix[MusicionType[i]][YEAR_DIC[str(MusicArr[i][1])]] += 1

    SimplePercentArr, SimpleGenre = GetSimpleMatrix(Num_Matrix)
    CreateLineChart(SimplePercentArr, SimpleGenre)

if __name__ == "__main__":
    ShowChart()