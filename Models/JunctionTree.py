#!/usr/bin/env python3
# coding=gbk
import networkx as nx

from pgmpy.models import ClusterGraph


class JunctionTree(ClusterGraph):


    def __init__(self, ebunch=None):
        super(JunctionTree, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.add_cliques=[]

    def add_edge(self, u, v, **kwargs):

        if u in self.nodes() and v in self.nodes() and nx.has_path(self, u, v):
            raise ValueError(
                f"Addition of edge between {str(u)} and {str(v)} forms a cycle breaking the properties of Junction Tree"
            )

        super(JunctionTree, self).add_edge(u, v, **kwargs)

    def check_model(self):

        if not nx.is_connected(self):
            raise ValueError("The Junction Tree defined is not fully connected.")

        return super(JunctionTree, self).check_model()

    def copy(self):

        copy = JunctionTree(self.edges())
        copy.add_nodes_from(self.nodes())
        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            copy.add_factors(*factors_copy)
        return copy
