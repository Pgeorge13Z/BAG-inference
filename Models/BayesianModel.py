# coding=gbk
from collections import defaultdict
import logging
from operator import mul
from functools import reduce

import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from pgmpy.base import DAG
from pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.independencies import Independencies
from Models.MarkovModelC import MarkovModel


class BayesianModel(DAG):


    def __init__(self, ebunch=None):
        super(BayesianModel, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_edge(self, u, v, **kwargs):

        if u == v:
            raise ValueError("Self loops are not allowed.")
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                "Loops are not allowed. Adding the edge from (%s->%s) forms a loop."
                % (u, v)
            )
        else:
            super(BayesianModel, self).add_edge(u, v, **kwargs)

    def remove_node(self, node):

        affected_nodes = [v for u, v in self.edges() if u == node]

        for affected_node in affected_nodes:
            node_cpd = self.get_cpds(node=affected_node)
            if node_cpd:
                node_cpd.marginalize([node], inplace=True)

        if self.get_cpds(node=node):
            self.remove_cpds(node)
        super(BayesianModel, self).remove_node(node)

    def remove_nodes_from(self, nodes):

        for node in nodes:
            self.remove_node(node)

    def add_cpds(self, *cpds):

        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor)):
                raise ValueError("Only TabularCPD or ContinuousFactor can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):

        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            for cpd in self.cpds:
                if cpd.variable == node:
                    return cpd
            else:
                return None
        else:
            return self.cpds

    def remove_cpds(self, *cpds):

        for cpd in cpds:
            if isinstance(cpd, str):
                cpd = self.get_cpds(cpd)
            self.cpds.remove(cpd)

    def get_cardinality(self, node=None):


        if node:
            return self.get_cpds(node).cardinality[0]
        else:
            cardinalities = defaultdict(int)
            for cpd in self.cpds:
                cardinalities[cpd.variable] = cpd.cardinality[0]
            return cardinalities

    def check_model(self):

        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
        return True

    def to_markov_model(self):

        moral_graph = self.moralize()  # 道德化
        mm = MarkovModel(moral_graph.edges())   # 建立markov模型
        mm.add_factors(*[cpd.to_factor() for cpd in self.cpds])

        return mm

    def to_junction_tree(self,heuristic,order=None):
        # 先转化为Markov模型，在转化为JunctionTree
        mm = self.to_markov_model()
        return mm.to_junction_tree(order=order,heuristic=heuristic)

    def fit(
        self, data, estimator=None, state_names=[], complete_samples_only=True, **kwargs
    ):


        from pgmpy.estimators import (
            MaximumLikelihoodEstimator,
            BayesianEstimator,
            BaseEstimator,
        )

        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        _estimator = estimator(
            self,
            data,
            state_names=state_names,
            complete_samples_only=complete_samples_only,
        )
        cpds_list = _estimator.get_parameters(**kwargs)
        self.add_cpds(*cpds_list)

    def predict(self, data, n_jobs=-1):

        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        data_unique = data.drop_duplicates()
        missing_variables = set(self.nodes()) - set(data_unique.columns)
        #         pred_values = defaultdict(list)
        pred_values = []

        # Send state_names dict from one of the estimated CPDs to the inference class.
        model_inference = VariableElimination(self)
        pred_values = Parallel(n_jobs=n_jobs)(
            delayed(model_inference.map_query)(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            for index, data_point in tqdm(
                data_unique.iterrows(), total=data_unique.shape[0]
            )
        )

        df_results = pd.DataFrame(pred_values, index=data_unique.index)
        data_with_results = pd.concat([data_unique, df_results], axis=1)
        return data.merge(data_with_results, how="left").loc[:, missing_variables]

    def predict_probability(self, data):

        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for index, data_point in data.iterrows():
            full_distribution = model_inference.query(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            states_dict = {}
            for var in missing_variables:
                states_dict[var] = full_distribution.marginalize(
                    missing_variables - {var}, inplace=False
                )
            for k, v in states_dict.items():
                for l in range(len(v.values)):
                    state = self.get_cpds(k).state_names[k][l]
                    pred_values[k + "_" + str(state)].append(v.values[l])
        return pd.DataFrame(pred_values, index=data.index)

    def get_factorized_product(self, latex=False):
        # TODO: refer to IMap class for explanation why this is not implemented.
        pass

    def is_imap(self, JPD):

        if not isinstance(JPD, JointProbabilityDistribution):
            raise TypeError("JPD must be an instance of JointProbabilityDistribution")
        factors = [cpd.to_factor() for cpd in self.get_cpds()]
        factor_prod = reduce(mul, factors)
        JPD_fact = DiscreteFactor(JPD.variables, JPD.cardinality, JPD.values)
        if JPD_fact == factor_prod:
            return True
        else:
            return False

    def copy(self):

        model_copy = BayesianModel()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def get_markov_blanket(self, node):

        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.remove(node)
        return list(blanket_nodes)