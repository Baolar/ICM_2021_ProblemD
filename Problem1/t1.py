# 第一问代码
import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.setrecursionlimit(9000000)
node_numbers = 5700 # 预设的最大的结点个数
dic = dict() # 字典，查找原id对应离散化的编号
node_max = -1 # 离散化后最大编号点
G = [[0] * node_numbers for _ in range(node_numbers)] # 图的邻接矩阵表示法
# 注意！！！！ 最大结点编号一定是node_max，不能是len(G)!!!!!!!
vis = [0] * node_numbers # 仅限dfs用
indegree = [0] * node_numbers
outdegree = [0] * node_numbers
DIC = dict() # 字典，查找离散化后编号对应原id
_max = 0 # 最大递归层数
t = 0 # 统计树的节点个数专用
def data_process():
    df = pd.read_csv('influence_data.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
    df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

    X = df[["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7","Col8"]]  # 抽取前七列作为训练数据的各属性值
    X = np.array(X)

    return X

def create_g(arr):
    global dic
    global G
    global node_max
    global indegree

    for i in range(len(arr)):
        influencer_id           = arr[i][0]
        influencer_name	        = arr[i][1]
        influencer_main_genre   = arr[i][2]
        influencer_active_start = arr[i][3]
        follower_id	            = arr[i][4]
        follower_name           = arr[i][5]
        follower_main_genre	    = arr[i][6]
        follower_active_start   = arr[i][7]
        
        if influencer_id not in dic:   
            node_max += 1                  
            dic[influencer_id] = node_max   
            DIC[node_max] = arr[i][0:4]

        if follower_id not in dic:
            node_max += 1
            dic[follower_id] = node_max
            DIC[node_max] = arr[i][4:8]

        G[dic[influencer_id]][dic[follower_id]] = 1   
        indegree[dic[follower_id]] += 1   
        outdegree[dic[influencer_id]] += 1

# 确保使用之前初始化了vis
def dfs(x, level):
    global t
    global vis
    global G
    global node_max
    global _max
    t += 1
    if level > _max:
        _max = level
    
    for i in range(node_max + 1):
        if G[x][i] == 1 and vis[i] == 0:
            vis[i] = 1
            dfs(i, level + 1)

# 从一个度为0的点开始dfs建树，把点和边都加在list里供绘图用
# 确保使用之前初始化了vis
nodelist = []
edgelist = []
def dfstree(x):
    global nodelist
    global edgelist
    vis[x] = 1
    nodelist.append(DIC[x][1])

    for i in range(node_max + 1):
        if G[x][i] == 1:
            edgelist.append((DIC[x][1], DIC[i][1]))
            if vis[i] == 0:
                dfstree(i)

def Print_Graph(x):
    dfstree(x)
    Gra = nx.DiGraph()
    Gra.add_nodes_from(nodelist)
    Gra.add_edges_from(edgelist)
    pos = nx.spring_layout(Gra)
    nx.draw(Gra, pos, with_labels=True, node_color='red', edge_color='gray', node_size=16, width=3.0)
    plt.show()
# 获取每一个结点的影响力（出度）
# 存在score[i]中并返回
# score[i]表示离散化后为i的结点的分数
def get_music_influence():
    score = [0] * (node_max + 1)

    for i in range(node_max + 1):
        cnt = 0
        for j in range(node_max + 1):
            if G[i][j] == 1:
                cnt += 1
        score[i] = cnt
    return score

# 以a为主元的排序
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
# 这个函数能把每一个入度为0的结点的dfs生成树的结点个数以及影响因子输出为文件
def outputf(): 
    global vis
    global cnt
    global indegree
    global t
    data=open("problem1_output.csv",'w+',encoding='utf-8') 

    score = get_music_influence() # 获取得分
    tot = 0
    for i in range(node_max + 1):
        if indegree[i] == 0:
            vis = [0] * node_numbers
            vis[i] = 1
            t = 0
            dfs(i, 0)
            print(str(i) + " " + str(DIC[i][0]) + " " + str(DIC[i][1]) + " " + str(DIC[i][2]) + " " + str(DIC[i][3]) + " " + str(t) + " " + str(score[i])) 
            print(str(i) + "," + str(DIC[i][0]) + "," + str(DIC[i][1]) + "," + str(DIC[i][2]) + "," + str(DIC[i][3]) + "," + str(t) + "," + str(score[i]), file=data) # 4428
            tot += 1
    data.close()
    print("tot = " + str(tot))
    vis = [0] * node_numbers
    t = 0

def scoref():
    score = get_music_influence()
    b = [_ for _ in range(len(score))]
    double_sort(score, b)
    data=open("problem1_scores.csv",'w+',encoding='utf-8') 

    for i in range(node_max + 1):
        print(str(b[i]) + " " + str(DIC[b[i]][0]) + " " + str(DIC[b[i]][1]) + " " + str(DIC[b[i]][2]) + " " + str(DIC[b[i]][3]) + " " + str(score[i])) 
        print(str(b[i]) + "," + str(DIC[b[i]][0]) + "," + str(DIC[b[i]][1]) + "," + str(DIC[b[i]][2]) + "," + str(DIC[b[i]][3]) + "," + str(score[i]), file=data) # 4428
    data.close()

if __name__ == "__main__":
    arr = data_process()
    create_g(arr)
    # scoref()
    # outputf()
    
    dfstree(dic[310455])
    Print_Graph(dic[310455])


    