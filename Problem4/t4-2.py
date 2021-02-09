import numpy as np
import pandas as pd
import sys
sys.setrecursionlimit(9000000)
#Default the maximum number of nodes
node_numbers = 6000

dic = dict()
node_max = -1 # 离散化后最大编号点
visit1 = [0] * node_numbers
visit2 = [0] * node_numbers
nodelist = []
edgelist = []
DIC = dict()
indegree = [0] * node_numbers
outdegree = [0] * node_numbers
# G第一维 离散化后u编号
# G第二维 离散化后v编号  u->v

# UTF-8编码格式csv文件数据读取
def DataProcess():
    df = pd.read_csv('influence_data.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

    X = df[["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7","Col8"]]  # 抽取前八列作为训练数据的各属性值
    X = np.array(X)

    return X

#创建图
def create_g(arr):
    global dic
    global templist
    # global  AllNum
    templist = [[] for _ in range(5610)]
    global G
    global node_max
    node_max = 0
    for i in range(len(arr)):
        influencer_id = arr[i][0]
        follower_id = arr[i][4]

        if influencer_id not in dic:  # 离散化
            nodelist.append(node_max)
            node_max += 1  # 所以每一个结点从0开始
            dic[influencer_id] = node_max
            DIC[node_max] = arr[i][0:4]
        if follower_id not in dic:
            nodelist.append(node_max)
            node_max += 1
            dic[follower_id] = node_max
            DIC[node_max] = arr[i][4:8]

        templist[dic[influencer_id]].append(dic[follower_id])
        t = (dic[influencer_id],dic[follower_id])
        indegree[dic[follower_id]] += 1  # 记录出度入度
        outdegree[dic[influencer_id]] += 1
        edgelist.append(t)
    return templist

Nodelist = []
def dfsGetMaxIndex(index):
    visit1[index] = 1
    global Nodelist
    Nodelist.append(index)
    for i in range(len(templist[index])):
        if (visit1[templist[index][i]] == 0):
            dfsGetMaxIndex(templist[index][i])

SimilarNum = 0
AllNum = 0
#对节点数最多的树进行dfs，求得所有边条数和所连节点类型相同的边条数
def dfs(index):
    global SimilarNum
    global AllNum
    visit2[index]=1
    for i in range(len(templist[index])):
        AllNum += 1
        if DIC[index][2] == DIC[templist[index][i]][2]:
            SimilarNum += 1
        if(visit2[templist[index][i]]==0):
            dfs(templist[index][i])

MaxNum = 0
MaxNumIndex = -1
#得到各孤立树中树的节点数最多的树的根节点
def GetMaxSimilarity():
    global MaxNum
    for i in range(1,node_max + 1):
        if visit1[i] == 0 and indegree[i]==0:
            visit1[i] = 1
            dfsGetMaxIndex(i)
            if(len(Nodelist)>MaxNum):
                MaxNum=len(Nodelist)
                MaxNumIndex=Nodelist[0]
                Nodelist.clear()
    dfs(MaxNumIndex)
    return SimilarNum/AllNum

#输出影响者和被影响者的相似度
if __name__ == "__main__":
    arr = DataProcess()
    create_g(arr)
    print(GetMaxSimilarity())