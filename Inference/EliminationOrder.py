# coding=gbk
import itertools
from abc import abstractmethod
from itertools import combinations
from tqdm import tqdm
import copy
import numpy as np

from pgmpy.models import BayesianModel,MarkovModel
from Models.BayesianModel import BayesianModel as BysModel
import networkx as nx
from functools import reduce

class BaseEliminationOrder:
    """
    Base class for finding elimination orders.
    """

    def __init__(self, model):
        """
        Init method for the base class of Elimination Orders.

        Parameters
        ----------
        model: BayesianModel instance
            The model on which we want to compute the elimination orders.
        """
        if not isinstance(model, (BayesianModel,BysModel)):
            raise ValueError("Model should be a BayesianModel instance")
        self.bayesian_model = model.copy()
        self.moralized_model = self.bayesian_model.moralize()
        self.Markov_model=MarkovModel(self.moralized_model.edges())
        self.Markov_model.add_factors(*[cpd.to_factor() for cpd in self.bayesian_model.cpds])

    @abstractmethod
    def cost(self, node):
        """
        The cost function to compute the cost of elimination of each node.
        This method is just a dummy and returns 0 for all the nodes. Actual cost functions
        are implemented in the classes inheriting BaseEliminationOrder.

        Parameters
        ----------
        node: string, any hashable python object.
            The node whose cost is to be computed.
        """
        return 0

    def get_elimination_order(self, nodes=None, show_progress=True):
        """
        Returns the optimal elimination order based on the cost function.
        The node having the least cost is removed first.

        Parameters
        ----------
        nodes: list, tuple, set (array-like)
            The variables which are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference.EliminationOrder import WeightedMinFill
        >>> model = BayesianModel([('c', 'd'), ('d', 'g'), ('i', 'g'),
        ...                        ('i', 's'), ('s', 'j'), ('g', 'l'),
        ...                        ('l', 'j'), ('j', 'h'), ('g', 'h')])
        >>> cpd_c = TabularCPD('c', 2, np.random.rand(2, 1))
        >>> cpd_d = TabularCPD('d', 2, np.random.rand(2, 2),
        ...                   ['c'], [2])
        >>> cpd_g = TabularCPD('g', 3, np.random.rand(3, 4),
        ...                   ['d', 'i'], [2, 2])
        >>> cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
        >>> cpd_s = TabularCPD('s', 2, np.random.rand(2, 2),
        ...                   ['i'], [2])
        >>> cpd_j = TabularCPD('j', 2, np.random.rand(2, 4),
        ...                   ['l', 's'], [2, 2])
        >>> cpd_l = TabularCPD('l', 2, np.random.rand(2, 3),
        ...                   ['g'], [3])
        >>> cpd_h = TabularCPD('h', 2, np.random.rand(2, 6),
        ...                   ['g', 'j'], [3, 2])
        >>> model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j,
        ...                cpd_l, cpd_h)
        >>> WeightedMinFill(model).get_elimination_order(['c', 'd', 'g', 'l', 's'])
        ['c', 's', 'l', 'd', 'g']
        >>> WeightedMinFill(model).get_elimination_order(['c', 'd', 'g', 'l', 's'])
        ['c', 's', 'l', 'd', 'g']
        >>> WeightedMinFill(model).get_elimination_order(['c', 'd', 'g', 'l', 's'])
        ['c', 's', 'l', 'd', 'g']
        """
        if nodes is None:
            nodes = self.bayesian_model.nodes()
        nodes = set(nodes)

        ordering = []
        if show_progress:
            pbar = tqdm(total=len(nodes))
            pbar.set_description("Finding Elimination Order: ")

        while nodes:
            scores = {node: self.cost(node) for node in nodes}
            min_score_node = min(scores, key=scores.get)
            ordering.append(min_score_node)
            nodes.remove(min_score_node)
            self.bayesian_model.remove_node(min_score_node)
            self.moralized_model.remove_node(min_score_node)

            if show_progress:
                pbar.update(1)
        return ordering

    def fill_in_edges(self, node):
        """
        Return edges needed to be added to the graph if a node is removed.

        Parameters
        ----------
        node: string (any hashable python object)
            Node to be removed from the graph.
        """
        return combinations(self.bayesian_model.neighbors(node), 2)


class WeightedMinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        Cost function for WeightedMinFill.
        The cost of eliminating a node is the sum of weights of the edges that need to
        be added to the graph due to its elimination, where a weight of an edge is the
        product of the weights, domain cardinality, of its constituent vertices.
        """
        edges = combinations(self.moralized_model.neighbors(node), 2)
        return sum(
            [
                self.bayesian_model.get_cardinality(edge[0])
                * self.bayesian_model.get_cardinality(edge[1])
                for edge in edges
            ]
        )


class MinNeighbors(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the number of neighbors it has in the
        current graph.
        """
        return len(list(self.moralized_model.neighbors(node)))


class MinWeight(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the product of weights, domain cardinality,
        of its neighbors.
        """
        return np.prod(
            [
                self.bayesian_model.get_cardinality(neig_node)
                for neig_node in self.moralized_model.neighbors(node)
            ]
        )

class MWH(BaseEliminationOrder):

    def cost(self,node):
        '''
        The cost of a elimination a node is the combine of MinWeight and H1.
        '''

        def _find_common_cliques(cliques_list):

            common = set([tuple(x) for x in cliques_list[0]])
            for i in range(1, len(cliques_list)):
                common = common & set([tuple(x) for x in cliques_list[i]])
            return list(common)

        def _find_size_of_clique(clique, cardinalities):

            return list(
                map(lambda x: np.prod([cardinalities[node] for node in x]), clique)
            )
        def _get_cliques_dict(node):

            graph_working_copy = nx.Graph(self.Markov_model.edges())
            neighbors = list(graph_working_copy.neighbors(node))
            graph_working_copy.add_edges_from(itertools.combinations(neighbors, 2))
            clique_dict = nx.cliques_containing_node(
                graph_working_copy, nodes=([node] + neighbors)
            )
            graph_working_copy.remove_node(node)
            clique_dict_removed = nx.cliques_containing_node(
                graph_working_copy, nodes=neighbors
            )
            return clique_dict, clique_dict_removed

        def _dispersion_level(nodes,hierarchy=3):
            '''
            计算node的邻居节点的节点连线密度

                node: 目标计算节点
                step: 计算到该节点的step层临接节点
                return:返回到hierarchy为止的总的节点关联密度 edges/nodes

            '''
            if isinstance(nodes,str):
                nodes=[nodes]

            #print('=====================')
            #print('nodes:', nodes)

            dispersion={} # 迭代几次的密度
            for hery in range(1,hierarchy+1):
                if len(nodes)==0:
                    dispersion[hery]=[{},{}]
                    continue
                neg_nodes=[] ; neg_edges=[]
                neighbor_edges_calculate = lambda node : list(edge for edge in self.moralized_model.edges if node in edge)
                neighbor_nodes_calculate = lambda node: list(self.moralized_model.neighbors(node))

                for node in nodes:
                    neg_nodes.extend(neighbor_nodes_calculate(node))
                    neg_edges.extend(neighbor_edges_calculate(node))

                neg_nodes=set(neg_nodes)
                neg_edges=set(neg_edges)

                #print('nodes--sec:',nodes)
                #print('neg_nodes:', neg_nodes)
                #print('neg_edges:', neg_edges)

                ''' dispersion[迭代次数]=[延伸总节点数,延伸总边数] '''
                dispersion[hery]=[set(nodes),neg_edges]

                #print('Dispersion_level:', len(neg_edges) / len(nodes))
                #print('dispersion:',dispersion)
                nodes = neg_nodes

            all_nodes = []
            all_edges = []
            
            for hery in range(1+1,hierarchy+1):
                all_nodes.extend(dispersion[hery][0])
                all_edges.extend(dispersion[hery][1])
            all_nodes=set(all_nodes)
            all_edges=set(all_edges)

            #print('all_nodes:',all_nodes)
            #print('all_edges:',all_edges)

            ''' 
            dis:all_edges/all_nodes->网络总边-节点密度
            dis1:计算节点的一次临接网络总边-节点密度
            '''
            try:
                #dis=len(dispersion[hierarchy][1])/len(dispersion[hierarchy][0])
                dis= len(all_edges)/len(all_nodes)
                dis1=len(dispersion[1][1])/len(dispersion[1][0])
                #print('dispersion:',dis)
            except:
                dis=0.0
                dis1=0.0

            return dis,dis1



        cardinalities = self.Markov_model.get_cardinality()

        MWcost=np.prod(
            [
                self.bayesian_model.get_cardinality(neig_node)
                for neig_node in self.moralized_model.neighbors(node)
            ]
        )

        clique_dict, clique_dict_removed = _get_cliques_dict(node)

        S= _find_size_of_clique(
                        _find_common_cliques(list(clique_dict_removed.values())),
                        cardinalities,
                    )[0]


        ''' 模糊评价法 '''
        dis,dis1=_dispersion_level(node)

        return MWcost,S/cardinalities[node],dis,dis1

    def get_elimination_order(self, nodes=None, show_progress=True):

        def turn_to_level(scores):
            '''将MinWeight 和 H1 启发式算法得出代价进行评级（用非整数次秩和比）'''
            scores_level = copy.deepcopy(scores)
            MWcost=[scores_level[node][0] for node in scores_level]
            Scost=[scores_level[node][1] for node in scores_level]
            DIScost=[scores_level[node][2] for node in scores_level]
            NEcost=[scores_level[node][3] for node in scores_level]

            # 非整数次秩和比法
            for node in scores_level:
                if max(MWcost)==min(MWcost):
                    MW_level=1
                else:
                    MW_level=1+ (len(scores_level)-1) * (scores_level[node][0]-min(MWcost))/(max(MWcost)-min(MWcost))

                if max(Scost)==min(Scost):
                    S_level=1
                else:
                    S_level=1+ (len(scores_level)-1) * (scores_level[node][1]-min(Scost))/(max(Scost)-min(Scost))

                if max(DIScost)==min(DIScost):
                    DIS_level=0
                else:
                    #DIS_level=1+ (len(scores_level)-1) * (scores_level[node][2]-min(DIScost))/(max(DIScost)-min(DIScost))
                    DIS_level = (scores_level[node][2] - min(DIScost)) / (max(DIScost) - min(DIScost))

                if max(NEcost) == min(NEcost):
                    NE_level = 0
                else:
                    NE_level =  (scores_level[node][3] - min(NEcost)) / (
                                max(NEcost) - min(NEcost))

                scores_level[node]=(MW_level,S_level,DIS_level,NE_level)
            return scores_level

        def evaluation(scores_level,scores):
            scores_fin={}
            for node in scores_level:
                # MW,S,dis
                # print('MW-----H1-----DIS')
                # print(scores_level[node][0],scores_level[node][1],scores_level[node][2])
                #scores_fin[node] = scores_level[node][0] + scores_level[node][1]+pow(scores_level[node][1] +scores_level[node][0], scores_level[node][3] )
                scores_fin[node] = scores_level[node][0] + scores_level[node][1]+pow(scores_level[node][1] , scores_level[node][3] )+ pow(scores_level[node][0],scores_level[node][2])
            return scores_fin


        if nodes is None:
            nodes = self.bayesian_model.nodes()
        nodes = set(nodes)

        ordering = []

        if show_progress:
            pbar = tqdm(total=len(nodes))
            pbar.set_description("Finding Elimination Order: ")

        while nodes:
            scores = {node: self.cost(node) for node in nodes}   # scores: node:(MWcost,H1cost,Dispersion)
            scores_level=turn_to_level(scores)
            # print('scores_level:',scores_level)
            scores_fin=evaluation(scores_level,scores)

            min_score_node = min(scores_fin, key=scores_fin.get)
            ordering.append(min_score_node)
            nodes.remove(min_score_node)
            self.bayesian_model.remove_node(min_score_node)
            self.moralized_model.remove_node(min_score_node)

            if show_progress:
                pbar.update(1)
        return ordering

class MinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the number of edges that need to be added
        (fill in edges) to the graph due to its elimination
        """
        return len(list(self.fill_in_edges(node)))
