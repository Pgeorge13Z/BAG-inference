# coding=gbk
import numpy as np
from Models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from itertools import combinations  # 排列 组合
import networkx as nx
import copy
import random
import matplotlib.pyplot as plt

def show_bn(edges):
    from graphviz import Digraph
    node_attr = dict(
        style='filled',
        # shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2'
    )

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    for a,b in edges:
        dot.edge(a,b)

    dot.view(cleanup=True,directory='/Users/apple/PycharmProjects/Algorithm_Repetition/Digraph',filename='try')
    return dot

# 攻击图可视化生成
def showBN(model,filename='BGA',isDGA=True,save=False):
    '''传入BayesianModel对象，调用graphviz绘制结构图，jupyter中可直接显示'''
    from graphviz import Digraph,Graph

    node_attr=dict(
        style='filled',
        #shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2'
    )

    if isDGA:
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    else:
        dot = Graph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen=set()
    edges=model.edges()
    for a,b in edges():
        dot.edge(a,b)
    if save:
        dot.view(cleanup=True,directory='/Users/apple/PycharmProjects/Algorithm_Repetition/Digraph',filename=filename)
    dot.view()
    return dot

def getCPD_OR(evidence_values_dict,evidence_cpd_dict):
    flag=-1
    n=1
    cpd = 0
    evidences={}
    for e,v in evidence_values_dict.items():
        if v=='1':
            evidences.update({e:v})
    if len(evidences)==0:
        cpd=0

    # print('evidences: ', evidences)
    # print('evidence_values: ', evidence_values_dict)

    for i in range(len(evidences)):
        evidences_temp=dict(zip(list(range(len(evidences))),evidences.keys()))
        ecombs=list(combinations(list(range(len(evidences))),n))
        # print('ecombs:',ecombs)
        for ecomb in ecombs:
            cpd_one=1
            for e in ecomb:
                cpd_one*=(evidence_cpd_dict[evidences_temp[e]])
            cpd+=(-1)*flag*cpd_one

        n+=1
        flag*=(-1)
    # print('cpds: ',[1-cpd,cpd])
    return [1-cpd,cpd]

# and型条件概率计算
def getCPD_AND(evidence_values_dict,evidence_cpd_dict):
    evidences = [e for e in evidence_values_dict.values() if e == '1']
    ecombs=combinations(evidences,1)
    cpd=1

    for e in ecombs:
        cpd*=evidence_cpd_dict[e]
    return [1-cpd,cpd]

# 节点间条件概率计算
def Values_Calulate(evidence,cpd,mode):

    evidence_cpd_dict=dict(zip(evidence,cpd))
    # print(evidence_cpd_dict)
    values=[]
    for eindex in range(2**len(evidence)):
        evidence_values=list(bin(eindex)[2:])
        # print('evidence: ',evidence)
        evidence_values_dict=dict(zip(evidence,evidence_values))
        if mode=='OR':
            value=getCPD_OR(evidence_values_dict,evidence_cpd_dict)
            values.append(value)
        elif mode=='AND':
            value=getCPD_AND(evidence_values_dict,evidence_cpd_dict)
            values.append(value)
        else:
            raise (Exception,"Invalid mode")
    values=np.array(values).T
    return values

# edges攻击成功概率
def getAttackProbility(node_list,model_edges,AttackProbability,mode):
    model_acctck_pro=[]

    for node in node_list.keys():
        parents=[];parents_cpd=[]
        for einfo in model_edges:
            if einfo[1]==node:
                parents.append(einfo[0])
                parents_cpd.append(AttackProbability[model_edges.index(einfo)])
        model_acctck_pro.append([parents,node,parents_cpd,mode])
    return model_acctck_pro


def connect(model_edges,all_nodes,avaliable_cnodes,Ppnodes,m,num_attack):
    edges = []
    for cnode in model_edges.keys():
        for pnode in model_edges[cnode]:
            edges.append((pnode, cnode))

    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(all_nodes)

    while len(Ppnodes)>num_attack:
        while len(avaliable_cnodes) > 0:

            random.shuffle(avaliable_cnodes)
            random.shuffle(all_nodes)

            cnode = avaliable_cnodes[0]
            size = np.random.randint(1, m + 1)

            index = 0
            if size>=len(all_nodes):
                size=len(all_nodes)-1
            #print('size:',size)

            while size:
                #print('index:',index)
                pnode = all_nodes[index]
                #print('   pnode:', pnode, '  cnode:', cnode)
                if pnode == cnode:
                    index += 1
                    pnode = all_nodes[index]

                model_edges[cnode].add(pnode)
                if pnode in avaliable_cnodes:
                    avaliable_cnodes.remove(pnode)

                size -= 1
                index += 1

            avaliable_cnodes.remove(cnode)


        Ppnodes = list(filter(lambda x: len(list(model_edges[x])) == 0, all_nodes))
        avaliable_cnodes=copy.deepcopy(Ppnodes)
        all_nodes=copy.deepcopy(Ppnodes)


def connect_cliques(all_nodes,avaliable_cnodes,model_edges,m):
    while len(avaliable_cnodes) > 0:
        random.shuffle(avaliable_cnodes)
        random.shuffle(all_nodes)

        cnode = avaliable_cnodes[0]
        size = np.random.randint(1, m + 1)

        index = 0
        if size >= len(all_nodes):
            size = len(all_nodes) - 1

        while size:
            pnode = all_nodes[index]
            if pnode == cnode:
                index += 1
                pnode = all_nodes[index]

            model_edges[cnode].add(pnode)
            if pnode in avaliable_cnodes:
                avaliable_cnodes.remove(pnode)

            size -= 1
            index += 1

        avaliable_cnodes.remove(cnode)

def Create_Pse(m,n_start,n_end,num_attack):
    import random

    model_edges={}
    all_nodes=list(str(node) for node in list(range(n_start,n_end+1)))
    avaliable_cnodes = list(str(node) for node in list(range(n_start,n_end+1)))

    for node in range(n_start,n_end+1):
        model_edges.update({str(node): set()})

    Ppnodes = list(filter(lambda x: len(list(model_edges[x])) == 0, all_nodes))

    connect(model_edges,all_nodes,avaliable_cnodes,Ppnodes,m,num_attack)

    edges=[]
    for cnode in model_edges.keys():
        for pnode in model_edges[cnode]:
            edges.append((pnode,cnode))


    node_evidence = [random.randint(0,1) for _ in range(n_end-n_start+1)]
    all_nodes=[str(node) for node in list(range(n_start,n_end+1))]  # 节点编号恢复从小到大顺序排序
    node_evidence=dict(zip(all_nodes,node_evidence))

    model_edges=edges
    AttackProbability=np.random.rand(len(model_edges))

    return node_evidence,model_edges,AttackProbability

def Create_Pse2(m,n_start,n_end,num_attack):
    import random

    model_edges={}
    all_nodes=list(str(node) for node in list(range(n_start,n_end+1)))
    avaliable_cnodes = list(str(node) for node in list(range(n_start,n_end+1)))

    for node in range(n_start,n_end+1):
        model_edges.update({str(node): set()})

    Ppnodes = list(filter(lambda x: len(list(model_edges[x])) == 0, all_nodes))

    edges = []
    for cnode in model_edges.keys():
        for pnode in model_edges[cnode]:
            edges.append((pnode, cnode))

    def connect_cliques1(edges, all_nodes, avaliable_cnodes, Ppnodes, m, num_attack):
        #print('edges:',edges)

        G = nx.Graph()
        G.add_edges_from(edges)
        G.add_nodes_from(all_nodes)
        subGraph = list(nx.connected_components(G))
        #print('subGraph len:', len(subGraph))

        while len(Ppnodes) > num_attack or len(subGraph)>1:
            while len(avaliable_cnodes) > 0:

                random.shuffle(avaliable_cnodes)
                random.shuffle(all_nodes)

                cnode = avaliable_cnodes[0]
                size = np.random.randint(1, m + 1)

                index = 0
                if size >= len(all_nodes):
                    size = len(all_nodes) - 1
                # print('size:',size)

                while size:
                    # print('index:',index)
                    pnode = all_nodes[index]
                    # print('   pnode:', pnode, '  cnode:', cnode)
                    if pnode == cnode:
                        index += 1
                        pnode = all_nodes[index]

                    edges.append((pnode,cnode))
                    if pnode in avaliable_cnodes:
                        avaliable_cnodes.remove(pnode)

                    size -= 1
                    index += 1

                avaliable_cnodes.remove(cnode)

            Ppnodes = list(filter(lambda x: len(list(edge for edge in edges if x==edge[1])) == 0, all_nodes))
            #print('edges:',edges)
            #print('Ppnodes:',Ppnodes)
            avaliable_cnodes = copy.deepcopy(Ppnodes)
            all_nodes = copy.deepcopy(Ppnodes)

            G = nx.Graph()
            G.add_edges_from(edges)
            G.add_nodes_from(all_nodes)
            subGraph = list(nx.connected_components(G))

    connect_cliques1(edges,all_nodes,avaliable_cnodes,Ppnodes,m,num_attack)

    node_evidence = [random.randint(0,1) for _ in range(n_end-n_start+1)]
    #all_nodes=[str(node) for node in list(range(n_start,n_end+1))]  # 节点编号恢复从小到大顺序排序
    node_evidence=dict(zip(all_nodes,node_evidence))

    model_edges=edges
    AttackProbability=np.random.rand(len(model_edges))

    return node_evidence,model_edges,AttackProbability

def Create_Cluster2(m,n,num_clique,num_attacker):
    '''
    :param m: 最大父节点数
    :param n: 每个簇的节点大小
    :param num_clique: 簇的数量
    :param num_attacker: 攻击者数量
    '''
    global color
    node_evidence_record=[]
    model_edges_record=[]

    DGA = np.zeros((n * num_clique, n * num_clique))
    for clique_num in range(num_clique):
        node_evidence,model_edges,_=Create_Pse(m,n*clique_num+1,n*(clique_num+1),num_attacker)
        #print('node_evdence:::',node_evidence)
        #print(len(node_evidence))
        for edge in model_edges:
            DGA[int(edge[0])-1][int(edge[1])-1]=1

        node_evidence_record.append(node_evidence)
        model_edges_record.extend(model_edges)

    #print('node_evidence_record:',node_evidence_record)

    #print('num_clique:',num_clique)

    for clique in node_evidence_record:
        for clique_add_to in node_evidence_record:
            if clique==clique_add_to:
                continue

            while (1):
                pnode=random.choice(list(clique.keys()))
                cnode=random.choice(list(clique_add_to.keys()))

                if (pnode,cnode) in model_edges_record:  # 已是父母节点
                    continue
                elif (cnode,pnode) not in model_edges_record:  # 也不是选定节点的父母节点
                    edges=copy.deepcopy(model_edges_record)
                    edges.append((pnode,cnode))
                    G=nx.DiGraph(edges)

                    try:
                        nx.find_cycle(G,orientation='original')
                    except:
                        # print(str(pnode)+'->'+str(node))
                        model_edges_record.append((pnode,cnode))
                        break


    node_evidence={}
    for node_ev in node_evidence_record:
        node_evidence.update(node_ev)
    AttackProbability = np.random.rand(len(model_edges_record))
    #print('len()', AttackProbability)
    print('node_evidences:',len(list(node_evidence.keys())))

    return node_evidence,model_edges_record,AttackProbability


def Create_Cluster(m,n,num_clique,num_attacker):
    '''
    :param m: 最大父节点数
    :param n: 每个簇的节点大小
    :param num_clique: 簇的数量
    :param num_attacker: 攻击者数量
    '''
    node_evidence_record=[]
    model_edges_record=[]

    DGA = np.zeros((n * num_clique, n * num_clique))
    for clique_num in range(num_clique):
        node_evidence,model_edges,_=Create_Pse(m,n*clique_num+1,n*(clique_num+1),num_attacker)
        #print('node_evdence:::',node_evidence)
        #print(len(node_evidence))

        node_evidence_record.append(node_evidence)
        model_edges_record.extend(model_edges)

    node_evidence = {}
    for node_ev in node_evidence_record:
        node_evidence.update(node_ev)


    def cliques_connect(model_edges_record,node_evidence):
        G = nx.Graph()
        G.add_edges_from(model_edges_record)
        G.add_nodes_from(node_evidence.keys())
        subGraph = list(nx.connected_components(G))
        print('subGraph len:',len(subGraph))

        while len(subGraph)>1:

            pclique_num = random.randint(0, len(subGraph) - 1)
            cclique_num = random.randint(0, len(subGraph) - 1)

            if pclique_num == cclique_num:
                continue

            #print('subGraph:', len(subGraph))
            #print('clique_r/clique_add_to_r:', cclique_num, pclique_num)

            clique = list(subGraph[pclique_num])
            clique_add_to = list(subGraph[cclique_num])

            while (1):
                pnode = random.choice(clique)
                cnode = random.choice(clique_add_to)

                if (pnode, cnode) in model_edges_record:  # 已是父母节点
                    continue
                elif (cnode, pnode) not in model_edges_record:  # 也不是选定节点的父母节点
                    edges = copy.deepcopy(model_edges_record)
                    edges.append((pnode, cnode))
                    G = nx.DiGraph(edges)

                    try:
                        nx.find_cycle(G, orientation='original')
                    except:
                        # print(str(pnode)+'->'+str(node))
                        model_edges_record.append((pnode, cnode))
                        break
            G = nx.Graph()
            G.add_edges_from(model_edges_record)
            G.add_nodes_from(node_evidence.keys())
            subGraph = list(nx.connected_components(G))


    cliques_connect(model_edges_record,node_evidence)

    AttackProbability = np.random.rand(len(model_edges_record))


    #print('len()', AttackProbability)
    print('node_evidences:',len(list(node_evidence.keys())))

    return node_evidence,model_edges_record,AttackProbability

def exsample(show_pic=False):
    # 节点的状态
    node_evidence = {'A': 1, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0}  # 0 1
    # 节点连接关系
    model_edges = [('A', 'D'),('A', 'B'),('A', 'C'), ('B', 'C'), ('C', 'E'), ('D', 'F'), ('E', 'F'), ('F', 'G')]
    # 攻击成功概率
    AttackProbability = [0.8, 0.8, 0.1, 0.9, 0.8, 0.9, 0.9, 0.1]

    model=BayesianModel(model_edges)
    if show_pic:
        dot=showBN(model,save=True)
    #dot = showBN(model)

    # 创建攻击信息  # 0 1
    model_attack_pro = getAttackProbility(node_evidence, model_edges, AttackProbability, 'OR')
    # [[[], 'A', [], 'OR'], [['A'], 'B', [0.8], 'OR'], [['A', 'B'], 'C', [0.1, 0.9], 'OR'], [['A'], 'D', [0.8], 'OR'], [['C'], 'E', [0.8], 'OR'], [['D', 'E'], 'F', [0.9, 0.9], 'OR'], [['F'], 'G', [0.1], 'OR']]

    CPD={}
    # add cpd
    for info in model_attack_pro:
        evidence=info[0]; node=info[1]; cpd=info[2]; mode=info[3]

        if len(evidence) == 0:  # 无父母节点
            CPD.update({node: TabularCPD(variable=node, variable_card=2, values=np.array([[0, 1.0]]).T)})
            continue

        values=Values_Calulate(evidence,cpd,mode)
        CPD.update({node:TabularCPD(variable=node,variable_card=2,values=values,evidence=evidence,evidence_card=[2]*len(evidence))})

    for node in CPD.keys():
        # print(CPD[node])
        model.add_cpds(CPD[node])
    model.check_model()

    #from Models.ExactInference import BeliefPropagation
    from Inference.BP import BeliefPropagation

    bp=BeliefPropagation(model,order=['A','B','C','D','E','F','G'])
    print('cliques:',bp.get_cliques())
    print('edges connecting cliques:',bp.get_clique_edges())
    clique_edges=[(str(edge_clique[0]),str(edge_clique[1])) for edge_clique in bp.get_clique_edges()]
    #show_bn(clique_edges)
    #bp = BeliefPropagation(model, order='H15')
    #bp.calibrate()   #校准
    #print('clique beliefs:',bp.get_clique_beliefs())
    #print('get cliques:',bp.get_cliques())

    bp.bp_query(variables=['B','A','C','A'],printable=True) # 该函数返回计数variables所需要的总时间，printable=True则输出各个variable的无条件概率图

    return bp

def Build_BGA_JT(node_evidence,model_edges,AttackProbability,order=None):    # order: random-随机顺序消除 / ['H1','H2',...]
                                                                              # ['MinWeight','WeightedMinFill','MinNeighbors','MinFill']

    model = BayesianModel(model_edges)
    #dot=showBN(model,filename='test',save=True)

    model_attack_pro=getAttackProbility(node_evidence,model_edges,AttackProbability,'OR')
    CPD = {}

    #print('model_attack_pro',model_attack_pro,len(model_attack_pro))
    for info in model_attack_pro:
        evidence=info[0]; node=info[1]; cpd=info[2]; mode=info[3]

        if len(evidence) == 0:  # 无父母节点
            CPD.update({node: TabularCPD(variable=node, variable_card=2, values=np.array([[0, 1.0]]).T)})
            continue

        values=Values_Calulate(evidence,cpd,mode)
        CPD.update({node:TabularCPD(variable=node,variable_card=2,values=values,evidence=evidence,evidence_card=[2]*len(evidence))})

    for node in CPD.keys():
        # print(CPD[node])

        model.add_cpds(CPD[node])
    model.check_model()

    from Inference.BP import BeliefPropagation
    bp=BeliefPropagation(model,order=order)
    # bp.calibrate()   #校准
    return bp


if __name__=="__main__":
    #exsample()
    m = 3  # 最大亲本数
    n = 20  # 网络总节点数
    num_attacker = int(n/3) # 攻击者数量

    color=np.zeros(n*3)
    isDGA=1
    exsample()
    #Creat_Pse(m,n)

    node_evidence, model_edges, AttackProbability=Create_Pse(m,1,n,num_attacker)
    bp=Build_BGA_JT(node_evidence,model_edges,AttackProbability,order='MinWeight')
    print('max_cliques:',len(bp.get_max_cliques()))
    print('time_caliberate:',bp.calibrate())