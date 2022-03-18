#!/usr/bin/env python3
# coding=gbk
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from pgmpy.base import UndirectedGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product
from pgmpy.independencies import Independencies


class MarkovModel(UndirectedGraph):

    def __init__(self, ebunch=None):
        super(MarkovModel, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []

    def add_edge(self, u, v, **kwargs):
        if u != v:
            super(MarkovModel, self).add_edge(u, v, **kwargs)
        else:
            raise ValueError("Self loops are not allowed")

    def add_factors(self, *factors):
        for factor in factors:
            if set(factor.variables) - set(factor.variables).intersection(
                set(self.nodes())
            ):
                raise ValueError("Factors defined on variable not in the model", factor)

            self.factors.append(factor)

    def get_factors(self, node=None):

        if node:
            if node not in self.nodes():
                raise ValueError("Node not present in the Undirected Graph")
            node_factors = []
            for factor in self.factors:
                if node in factor.scope():
                    node_factors.append(factor)
            return node_factors
        else:
            return self.factors

    def remove_factors(self, *factors):

        for factor in factors:
            self.factors.remove(factor)

    def get_cardinality(self, node=None):

        if node:
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    if node == variable:
                        return cardinality
        else:
            cardinalities = defaultdict(int)
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    cardinalities[variable] = cardinality
            return cardinalities

    def check_model(self):

        cardinalities = self.get_cardinality()
        for factor in self.factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                if cardinalities[variable] != cardinality:
                    raise ValueError(
                        "Cardinality of variable {var} not matching among factors".format(
                            var=variable
                        )
                    )
                if len(self.nodes()) != len(cardinalities):
                    raise ValueError("Factors for all the variables not defined")
            for var1, var2 in itertools.combinations(factor.variables, 2):
                if var2 not in self.neighbors(var1):
                    raise ValueError("DiscreteFactor inconsistent with the model.")
        return True

    def to_factor_graph(self):

        from pgmpy.models import FactorGraph

        factor_graph = FactorGraph()

        if not self.factors:
            raise ValueError("Factors not associated with the random variables.")

        factor_graph.add_nodes_from(self.nodes())
        for factor in self.factors:
            scope = factor.scope()
            factor_node = "phi_" + "_".join(scope)
            factor_graph.add_edges_from(itertools.product(scope, [factor_node]))
            factor_graph.add_factors(factor)

        return factor_graph

    def triangulate(self, heuristic='H2', order=None, inplace=False):

        self.check_model()

        if self.is_triangulated():
            if inplace:
                return
            else:
                return self

        graph_copy = nx.Graph(self.edges())
        edge_set = set()

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

            graph_working_copy = nx.Graph(graph_copy.edges())
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

        if not order:
            order = []

            cardinalities = self.get_cardinality()

            '''格式
            nodes: ['7', '1', '14', '15', '10', '2', '16', '20', '8', '17', '9', '4', '3', '5', '18', '6', '11', '12', '13', '19']
            cardinalities: defaultdict(<class 'int'>, {'1': 2, '7': 2, '14': 2, '2': 2, '3': 2, '4': 2, '9': 2, '5': 2, '6': 2, '15': 2, '10': 2, '8': 2, '11': 2, '12': 2, '13': 2, '20': 2, '16': 2, '17': 2, '18': 2, '19': 2})
            '''

            for index in range(self.number_of_nodes()):
                # S represents the size of clique created by deleting the
                # node from the graph
                S = {}
                # M represents the size of maximum size of cliques given by
                # the node and its adjacent node
                M = {}
                # C represents the sum of size of the cliques created by the
                # node and its adjacent node
                C = {}
                MW={}

                for node in set(graph_copy.nodes()) - set(order):
                    clique_dict, clique_dict_removed = _get_cliques_dict(node)
                    S[node] = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict_removed.values())),
                        cardinalities,
                    )[0]
                    common_clique_size = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict.values())), cardinalities
                    )
                    M[node] = np.max(common_clique_size)
                    C[node] = np.sum(common_clique_size)

                if index == 0:
                    pass

                if heuristic == "H1":
                    node_to_delete = min(S, key=S.get)

                elif heuristic == "H2":
                    S_by_E = { key: S[key] / cardinalities[key] for key in S }
                    node_to_delete = min(S_by_E, key=S_by_E.get)

                elif heuristic == 'H15':
                    node_to_delete = min(MW, key=MW.get)

                elif heuristic == "H3":
                    S_minus_M = { key: S[key] - M[key] for key in S }
                    node_to_delete = min(S_minus_M, key=S_minus_M.get)

                elif heuristic == "H4":
                    S_minus_C = { key: S[key] - C[key] for key in S }
                    node_to_delete = min(S_minus_C, key=S_minus_C.get)

                elif heuristic == "H5":
                    S_by_M = { key: S[key] / M[key] for key in S }
                    node_to_delete = min(S_by_M, key=S_by_M.get)

                else:
                    S_by_C = {key: S[key] / C[key] for key in S}
                    node_to_delete = min(S_by_C, key=S_by_C.get)

                order.append(node_to_delete)
        #print('elimination orders:',order)

        graph_copy = nx.Graph(self.edges())
        for node in order:
            for edge in itertools.combinations(graph_copy.neighbors(node), 2):
                graph_copy.add_edge(edge[0], edge[1])
                edge_set.add(edge)
            graph_copy.remove_node(node)

        if inplace:
            for edge in edge_set:
                self.add_edge(edge[0], edge[1])
            return self

        else:
            graph_copy = MarkovModel(self.edges())
            for edge in edge_set:
                graph_copy.add_edge(edge[0], edge[1])
            return graph_copy


    def to_junction_tree(self,heuristic,order=None):

        from Models.JunctionTree import JunctionTree

        # Check whether the model is valid or not
        self.check_model()

        # Triangulate the graph to make it chordal
        triangulated_graph = self.triangulate(order=order,heuristic=heuristic)

        # Find maximal cliques in the chordal graph
        cliques = list(map(tuple, nx.find_cliques(triangulated_graph)))
        # print('cliques:',cliques)

        # If there is only 1 clique, then the junction tree formed is just a
        # clique tree with that single clique as the node
        if len(cliques) == 1:
            clique_trees = JunctionTree()
            clique_trees.add_node(cliques[0])

        # Else if the number of cliques is more than 1 then create a complete
        # graph with all the cliques as nodes and weight of the edges being
        # the length of sepset between two cliques
        elif len(cliques) >= 2:
            complete_graph = UndirectedGraph()
            edges = list(itertools.combinations(cliques, 2))
            weights = list(map(lambda x: len(set(x[0]).intersection(set(x[1]))), edges))
            for edge, weight in zip(edges, weights):
                complete_graph.add_edge(*edge, weight=-weight)

            #print('complete graph:',complete_graph)
            # Create clique trees by minimum (or maximum) spanning tree method
            #print(nx.minimum_spanning_tree(complete_graph).edges())
            clique_trees = JunctionTree(
                nx.minimum_spanning_tree(complete_graph).edges()
            )

        # print('clique_trees.edges:',clique_trees.edges)
        # print('clique_trees.nodes:',clique_trees.nodes)

        add_edge={}   #  存放需要添加的节点信息  虚加节点：在树中相连节点
        self.add_cliques=[]  # 记录增加的clique
        '''
        for node in clique_trees.nodes:
            neighbors_edges=list(filter(lambda x:node in x,clique_trees.edges))
            if len(neighbors_edges)==1:
                #print('neighbors_edges:',neighbors_edges)
                #print('node:',{node},'neighbors_edges:',set(neighbors_edges[0]))
                #print('******:',set(neighbors_edges[0])-{node})
                clique_edge_to=set(neighbors_edges[0])-{node}
                #print('clique_edge_to:',list(clique_edge_to)[0])
                clique_add=tuple(set(node)-set(list(clique_edge_to)[0]))
                #print('clique_add:',clique_add)
                #print('clique node:',node)
                add_edge.update({clique_add:node})

        for clique_add in add_edge.keys():
            #clique_trees.add_node(clique_add)
            #clique_trees.add_edge(clique_add,add_edge[clique_add])
            #clique_trees.add_cliques.append(clique_add)
            pass
            #clique_trees.remove_node(clique_add)
            #print('clique_trees.nodes:+++++', clique_trees.nodes)
        '''

        # Check whether the factors are defined for all the random variables or not
        all_vars = itertools.chain(*[factor.scope() for factor in self.factors])
        #print('all_vars:',all_vars)  # 原图形的节点与其父母节点构成因子
        #print('factors:',self.factors)
        #for fac in self.factors:
        #    print('factor.scope',fac.scope())
        if set(all_vars) != set(self.nodes()):
            ValueError("DiscreteFactor for all the random variables not specified")

        # Dictionary stating whether the factor is used to create clique
        # potential or not
        # If false, then it is not used to create any clique potential
        is_used = {factor: False for factor in self.factors}


        for node in clique_trees.nodes():
            clique_factors = []
            for factor in self.factors:
                # If the factor is not used in creating any clique potential as
                # well as has any variable of the given clique in its scope,
                # then use it in creating clique potential

                # issubset() 方法用于判断集合的所有元素是否都包含在指定集合中，如果是则返回 True，否则返回 False。
                if not is_used[factor] and set(factor.scope()).issubset(node):
                    clique_factors.append(factor)
                    is_used[factor] = True

            # print('clique_factors:',clique_factors)
            '''
            clique_factors:
            (G:2,F:2)
            (F:2, D:2, E:2)
            (A:2),(B:2, A:2),(C:2, A:2, B:2),(D:2, A:2)
            (E:2, C:2)
            '''

            # To compute clique potential, initially set it as unity factor
            var_card = [self.get_cardinality()[x] for x in node]
            #print('node:',node)
            #print('var_card:',var_card)

            '''
            self.factors:('A'),('B','A'),('C', 'A', 'B'),('D', 'A'),('E', 'C'),
                         ('F', 'D', 'E'),('G', 'F')
            var_cardinality: defaultdict(<class 'int'>, {'A': 2, 'B': 2, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 2})
            ('A')-> var_card: [2, 2]
            ('C', 'A', 'B')-> var_card: [2, 2, 2]  -> var_card:一个factor里包含的每个元素的card数
            ...
            '''
            clique_potential = DiscreteFactor(
                node, var_card, np.ones(np.product(var_card))  #np.product: 做笛卡尔积
                ### PCA降维
            )
            #print('clique_potential:\n',clique_potential)

            # multiply it with the factors associated with the variables present
            # in the clique (or node)
            # Checking if there's clique_factors, to handle the case when clique_factors
            # is empty, otherwise factor_product with throw an error [ref #889]
            if clique_factors:
                #print('jnkn',clique_factors)
                clique_potential *= factor_product(*clique_factors)
            #print('jnkn', node,clique_potential)
            clique_trees.add_factors(clique_potential)
            #print('clique_trees_factors:\n',clique_potential)

        if not all(is_used.values()):
            raise ValueError(
                "All the factors were not used to create Junction Tree."
                "Extra factors are defined."
            )

        return clique_trees

    def markov_blanket(self, node):

        return self.neighbors(node)

    def get_local_independencies(self, latex=False):

        local_independencies = Independencies()

        all_vars = set(self.nodes())
        for node in self.nodes():
            markov_blanket = set(self.markov_blanket(node))
            rest = all_vars - set([node]) - markov_blanket
            try:
                local_independencies.add_assertions(
                    [node, list(rest), list(markov_blanket)]
                )
            except ValueError:
                pass

        local_independencies.reduce()

        if latex:
            return local_independencies.latex_string()
        else:
            return local_independencies

    def to_bayesian_model(self):

        from pgmpy.models import BayesianModel

        bm = BayesianModel()
        var_clique_dict = defaultdict(tuple)
        var_order = []

        # Create a junction tree from the markov model.
        # Creation of clique tree involves triangulation, finding maximal cliques
        # and creating a tree from these cliques
        junction_tree = self.to_junction_tree()

        # create an ordering of the nodes based on the ordering of the clique
        # in which it appeared first
        root_node = next(iter(junction_tree.nodes()))
        bfs_edges = nx.bfs_edges(junction_tree, root_node)
        for node in root_node:
            var_clique_dict[node] = root_node
            var_order.append(node)
        for edge in bfs_edges:
            clique_node = edge[1]
            for node in clique_node:
                if not var_clique_dict[node]:
                    var_clique_dict[node] = clique_node
                    var_order.append(node)

        # create a bayesian model by adding edges from parent of node to node as
        # par(x_i) = (var(c_k) - x_i) \cap {x_1, ..., x_{i-1}}
        for node_index in range(len(var_order)):
            node = var_order[node_index]
            node_parents = (set(var_clique_dict[node]) - set([node])).intersection(
                set(var_order[:node_index])
            )
            bm.add_edges_from([(parent, node) for parent in node_parents])
            # TODO : Convert factor into CPDs
        return bm

    def get_partition_function(self):

        self.check_model()

        factor = self.factors[0]
        factor = factor_product(
            factor, *[self.factors[i] for i in range(1, len(self.factors))]
        )
        if set(factor.scope()) != set(self.nodes()):
            raise ValueError("DiscreteFactor for all the random variables not defined.")

        return np.sum(factor.values)

    def copy(self):
        clone_graph = MarkovModel(self.edges())
        clone_graph.add_nodes_from(self.nodes())

        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            clone_graph.add_factors(*factors_copy)

        return clone_graph
