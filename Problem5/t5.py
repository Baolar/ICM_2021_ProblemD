import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False

#The list of music genres
genre = ["Vocal", "Stage & Screen", "Religious", "Reggae", "R&B;",
             "Pop/Rock", "New Age", "Latin", "Jazz", "International", "Unknown",
             "Folk", "Electronic", "Easy Listening", "Country", "Comedy/Spoken", "Classical",
             "Children's", "Blues", "Avant-Garde"]

#The list of music characteristic
characteristic = ["danceability","energy","valence","tempo","loudness","key"]

#The list of the years we need
year = ['1920','1930','1940','1950','1960','1970','1980','1990','2000','2010']

#The dictionary of genres and years
GENRE_DIC = dict()
YEAR_DIC = dict()

#Get the data we need:Normalized annual list of music characteristic/list of years
def GetReadyData():
    for i in range(len(genre)):
        GENRE_DIC[genre[i]] = i
    for i in range(len(year)):
        YEAR_DIC[year[i]] = i

    df = pd.read_csv('data_by_year.csv')
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8","Col9", "Col10", "Col11", "Col12", "Col13", "Col14"]  # 抽取前八列作为训练数据的各属性值
    X = df[["Col1", "Col2", "Col3", "Col4", "Col5", "Col6",  "Col8"]]
    X = np.array(X)
    MusicArr = []
    YearArr = []
    for i in range(len(X)):
        temp_year = (X[i][0]//10)*10
        if str(int(temp_year)) in YEAR_DIC.keys():
            temp_music = [X[i][1],X[i][2],X[i][3],X[i][4],X[i][5],X[i][6]]
            MusicArr.append(temp_music)
            YearArr.append(temp_year)
    MusicArr = (MusicArr - np.min(MusicArr, axis=0)) / (np.max(MusicArr, axis=0) - np.min(MusicArr, axis=0))
    return MusicArr,YearArr

#Merged the musics released in the same year
def MergeMusicArr(MusicArr,YearArr):
    temp_list = []
    temp_year = YearArr[0]
    MergeArr = []
    for i in range(len(MusicArr)):
        if YearArr[i] != temp_year:
            temp_merge = np.sum(temp_list,axis=0)
            temp_merge = np.divide(temp_merge, len(temp_list))
            MergeArr.append(temp_merge)
            temp_list = []
            temp_year = YearArr[i]
        temp_list.append(MusicArr[i])
    temp_merge = np.sum(temp_list, axis=0)
    temp_merge = np.divide(temp_merge, len(temp_list))
    MergeArr.append(temp_merge)
    return np.array(MergeArr)

#Get all the music
def GetReadyData_Music():
    for i in range(len(genre)):
        GENRE_DIC[genre[i]] = i
    for i in range(len(year)):
        YEAR_DIC[year[i]] = i

    df = pd.read_csv('full_music_data.csv')
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8","Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16","Col17", "Col18", "Col19"]  # 抽取前八列作为训练数据的各属性值
    X = df[["Col2","Col17"]]
    X = np.array(X)
    MusicArr = []
    for i in range(len(X)):
        temp_year = (X[i][1]//10)*10
        if str(temp_year) in YEAR_DIC.keys():
            MusicArr.append([X[i][0],temp_year])
    return MusicArr

#Get all of the music genres corresponding to the musician
def AllMusicionType():
    df = pd.read_csv('influence_data.csv')
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

    list1 = df[["Col1", "Col3"]]
    list2 = df[["Col5", "Col7"]]

    temp1 = np.array(list1)
    temp2 = np.array(list2)
    temp = np.concatenate((temp1,temp2),axis=0)

    MusicionType = np.array(list(set([tuple(t) for t in temp])))
    return MusicionType

#Get the special music genre corresponding to the musician
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

#Simplify the Num_Matrix
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

#Draw diagrams
def CreateLineChart(MergeArr,SimplePercentArr,SimpleGenre):
    fig,ax1 = plt.subplots()
    x = year
    for i in range(len(SimplePercentArr)):
        y = SimplePercentArr[i]
        plt.plot(x, y, marker='.')
    plt.legend(SimpleGenre, loc='lower right', bbox_to_anchor=(0.77, 0.2))
    plt.xlabel("The year")
    plt.ylabel('The Percentage')

    ax2 = ax1.twinx()
    for i in range(0,len(MergeArr[0])):
        y = MergeArr[:,i]
        plt.plot(x,y,':',marker='.')
    plt.legend(characteristic, loc='lower left', bbox_to_anchor=(0.77, 0.2))

    plt.title("The weight of characteristics/percentage of genres changed over time")
    plt.xlabel("The year")
    plt.ylabel("The weight")
    plt.show()

#Integrate the functions needed for drawing
def ShowChart():
    MusicArr = GetReadyData_Music()

    all_musicion_type = AllMusicionType().tolist()
    section = [i[0] for i in all_musicion_type]
    MusicionType, MusicArr = GetMusicionType(all_musicion_type, MusicArr, section, MusicArr)

    Num_Matrix = [[0] * (len(year)) for i in range((len(genre)))]
    for i in range(len(MusicArr)):
        Num_Matrix[MusicionType[i]][YEAR_DIC[str(MusicArr[i][1])]] += 1

    SimplePercentArr, SimpleGenre = GetSimpleMatrix(Num_Matrix)
    return SimplePercentArr, SimpleGenre

#Get the dictionary:{ArtistId:[genre,amount of music]}
def CreateADIC():
    ADIC = dict()
    df = pd.read_csv('influence_data.csv')
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]
    list1 = df[["Col1", "Col3"]]
    temp1 = np.array(list1)

    for row in temp1:
        if row[0] not in ADIC:
            ADIC[row[0]] = [GENRE_DIC[row[1]], 1]
        else:
            ADIC[row[0]][1] += 1
    return ADIC

#Get the set of songs released by a particular type of musician in a given year
def GetSpeicalMusic(y, Type,ADIC):
    ans = []
    df = pd.read_csv('full_music_data.csv')
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8","Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16","Col17", "Col18", "Col19"]  # 抽取前八列作为训练数据的各属性值
    X = df[["Col2","Col17"]]
    X = np.array(X)
    TMP = [] # id year preprocess
    for i in range(len(X)):
        temp = X[i][0].lstrip('[')
        temp = temp.rstrip(']')
        flag = True
        for _ in temp:
            if '0' > _ or _ > '9':
                flag = False
                break
        if flag == True:
            TMP.append([int(temp), X[i][1]])

    for m in TMP:
        if m[0] not in ADIC:
            continue
            print(m[0])
        if m[1] // 10 * 10 == y and ADIC[m[0]][0] == Type:
            ans.append(m[0])

    return list(set(ans))

# Sorting with 'a' as the principal element
def double_sort(a, b):
    for i in range(len(a)):
        for j in range(len(a) - i - 1):
            if a[j] < a[j+1]:
                t = a[j]
                a[j] = a[j + 1]
                a[j + 1] = t
                t = b[j]
                b[j] = b[j + 1]
                b[j + 1] = t


if __name__ == "__main__":
    #Print the chart of "The weight of characteristics/percentage of genres changed over time"
    MusicArr,YearArr = GetReadyData()
    MergeArr = MergeMusicArr(MusicArr,YearArr)
    SimplePercentArr, SimpleGenre = ShowChart()
    CreateLineChart(MergeArr,SimplePercentArr, SimpleGenre)

    #Get the set of songs released by pop artists in 1950
    ADIC = CreateADIC()
    l = GetSpeicalMusic(1950, 5,ADIC)
    scores = []
    for _ in l:
        scores.append(ADIC[_][1])
    double_sort(scores, l)
    #Take the top ten singers in the set
    for i in range(min(len(l), 10)):
        print(str(l[i]) + " " + str(scores[i]))