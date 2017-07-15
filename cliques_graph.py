import networkx as nx
from itertools import combinations
import numpy as np
import pandas as pd


class CliquesGraph():
    #
    def __init__(self, analyze_df, historical_expectation_object, expectation_connection_factor):
        """
        This Cliques object does two primary things:
        1) It generates the graph induced by analyze_df (the week under current examination), based on the expectations
        set by the historical_expectation_object.
        2) It allows for retrieval of selected cliques, based on certain filters (i.e. don't give me any cliques that
        include State=CA)

        :param analyze_df: A dataframe of the time period currently under examination
        :param historical_expectation_object: A trained HistoricalExpectationObject (see the expectation_objects
        file) containing the expected frequencies/counts for each combination of features you expect to see in the analyze_df
        :param expectation_connection_factor: A parameter to determine how large a cluster needs to be to meet our
        criteria of "anomalous"
        :return:
        """
        nx_graph = CliquesGraph.generate_graph(analyze_df, historical_expectation_object, expectation_connection_factor)
        ##Assumes =analyze_df contains fields for 'loan_state'
        self.data_df = analyze_df
        self.cliques = []
        self.graph = nx_graph

        for c in nx.find_cliques(nx_graph):
            subg = nx_graph.subgraph(c)
            clique_dict = dict()
            clique_dict['description'] = c
            clique_dict['cluster_weight'] = subg.size(weight='weight')
            if clique_dict['cluster_weight'] > 0:
                self.cliques.append(clique_dict)

    def get_cliques(self, filter_functions=()):
        """
        :param filter_functions: A list of boolean-returning lambda functions,
        all of which must be met for a cluster to be returned

        :return: A list of clique descriptions
        """
        out = []
        for clique in self.cliques:
            filter_results = [el(clique) for el in filter_functions]
            if np.all(filter_results):
                out.append(clique)
        return out

    def get_unioned_clique_data(self, filter_functions=()):
        """
        :param filter_functions: A list of boolean-returning lambda functions,
        all of which must be met for a cluster to be returned

        :return: The dataframe created by taking the union of all the feature intersections that make up the clusters
        """
        base_df = self.data_df.head()
        i = 0
        for clique in self.cliques:
            filter_results = [el(clique) for el in filter_functions]
            if np.all(filter_results):
                clique_data = query_subset(self.data_df, clique['description'])
                if i == 0:
                    base_df = clique_data
                else:
                    base_df = pd.concat([base_df, clique_data])
                i += 1
        return base_df.drop_duplicates(subset='user_id')

    @staticmethod
    def generate_graph(group_df, historical_expectation_object, expectation_factor):
        """

        :param group_df: The dataframe currently under analysis
        :param historical_expectation_object: A trained HistoricalExpectationsObject
        :param expectation_factor: The amount of total connection-weight a cluster needs to have to be considered sufficiently
        :return: A trained graph, where nodes are of the type Feature=VAL (i.e. State=CA, channel=SEARCH) and linkages
        are present between two category/values when the intersecting cell in that 2x2 crosstab is significantly above
        historical expectations

        """
        g = nx.Graph()
        for column_set in historical_expectation_object.expectations_map.keys():
            expectations = historical_expectation_object.expectations_map[column_set]
            left_col, right_col = column_set.split('~')
            group_df[left_col] = group_df[left_col].astype('category')
            group_df[right_col] = group_df[right_col].astype('category')

            empirical_counts = (group_df.groupby([left_col, right_col]).size().reset_index()).pivot(left_col, right_col, values=0)

            # Implicitly connect between values of same category when those values connect to anything else
            for tup_left in combinations(empirical_counts.index, 2):
                g.add_edge("%s=%s" % (left_col, tup_left[0]), "%s=%s" % (left_col, tup_left[1]), weight=0)

            for tup_right in combinations(empirical_counts.columns, 2):
                g.add_edge("%s=%s" % (right_col, tup_right[0]), "%s=%s" % (right_col, tup_right[1]), weight=0)
            #pdb.set_trace()
            amt_above_exp = expectations.amount_above_expectations(empirical_counts)
            # if left_col == 'br_first_two' or right_col == 'br_first_two' or left_col == 'leadsourcecategory' or right_col == 'leadsourcecategory':
            #     pdb.set_trace()
            for c1 in amt_above_exp.index:
                for c2 in amt_above_exp.columns:
                    if amt_above_exp.loc[c1, c2] > expectation_factor:
                        g.add_edge("%s=%s" % (left_col, c1), "%s=%s" % (right_col, c2), weight=amt_above_exp.loc[c1, c2])
        return g
