from math import exp
from numpy.core.numeric import False_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rpy2.robjects.packages import data
from sqlalchemy.engine import base
import seaborn as sns
import subprocess
import pickle
import os
import argparse
from bayes_factor import BayesFactor
from predicate_induction.data_type import Tabular
from predicate_induction.predicate import Conjunction
from predicate_induction.predicate_induction import BottomUp

class Pixal:
    """Main class for running PIXAL algorithm
    
    :param df: Data to run PIXAL on
    :type df: pd.DataFrame
    :param num_bins: Number of bins when converting numeric columns to ordinal
    :type num_bins: int
    :param num_points_per_bin: Number of points to include in each bin when converting numeric columns to ordinal
    :type num_points_per_bin: int
    """

    def __init__(self, data=None, dtypes=None, num_bins=15, num_points_per_bin=None, data_obj=None, frontier=None,
                       accepted=None, rejected=None, conditionally_accepted=None, base_predicates=None,
                       anomaly_feature=None, explanation_features=None):
        if data_obj is None:
            self.data_obj = Tabular(num_bins=num_bins, num_points_per_bin=num_points_per_bin)
            self.data_obj.extract(data, dtypes)
        else:
            self.data_obj = data_obj
        self.data_obj.convert_all(Conjunction.allowed_dtypes_map, Conjunction.allowed_dtypes)
        
        self.explanation_features = []
        if anomaly_feature is not None or explanation_features is not None:
            self.set_features(anomaly_feature, explanation_features, base_predicates, frontier, accepted, rejected)
        else:
            if frontier is None:
                self.frontier = []
            else:
                self.frontier = frontier
            if accepted is None:
                self.accepted = []
            else:
                self.accepted = accepted
            if rejected is None:
                self.rejected = []
            else:
                self.rejected = rejected
        if conditionally_accepted is None:
            self.conditionally_accepted = []
        else:
            self.conditionally_accepted = conditionally_accepted
        self.bf = BayesFactor(self.data_obj.dtypes)
        self.search = None

    def update_frontier_accepted_rejected(self):
        if self.search is not None:
            self.frontier = self.predicates_to_explanations(self.search.frontier)
            self.accepted = self.predicates_to_explanations(self.search.accepted)
            self.rejected = self.predicates_to_explanations(self.search.rejected)
        else:
            self.frontier = []
            self.accepted = []
            self.rejected = []

    def set_features(self, anomaly_feature=None, explanation_features=None, base_predicates=None, frontier=None, accepted=None, rejected=None):
        """Set the anomaly feature and/or explanation features.

        :param anomaly_feature: Name of the column to be used as the anomaly score
        :type anomaly_feature: str
        :param explanation_features: List of columns to be included in explanations
        :type explanation_features: list
        """

        if anomaly_feature is not None:
            self.score_f = lambda mask: self.bf.bayes_factor(self.data_obj.data, anomaly_feature, mask=mask, side='left')
            self.anomaly_feature = anomaly_feature
            self.anomaly_feature_dtype = self.data_obj.dtypes[self.anomaly_feature]
        if explanation_features is not None:
            if base_predicates is None:
                base_predicates = Conjunction.bottom_up_init(data_obj=self.data_obj, columns=explanation_features)
            self.explanation_features = explanation_features
        if base_predicates is not None:
            self.search = BottomUp(self.data_obj.data, base_predicates, self.score_f, frontier, accepted, rejected)
        self.update_frontier_accepted_rejected()

#     def predicate_to_explanation(self, predicate):
#         """Convert the given predicate instance to an explanation instance.

#         :param predicate: Predicate to be converted
#         :type predicate: Predicate
#         :return: Converted explanation
#         :rtype: AnomalyExplanation
#         """

#         return AnomalyExplanationOld(self.anomaly_feature, self.data_obj.original_data, self.data_obj.original_dtypes, predicate)

#     def predicates_to_explanations(self, predicates):
#         if type(predicates) == dict:
#             return {k: self.predicates_to_explanations(v) for k, v in predicates.items()}
#         else:
#             return [self.predicate_to_explanation(predicate) for predicate in predicates]

    def explanation_to_dict(self, explanation, include_adjacent=True, predicates_find_adjacent=None):
        res = {
            'anomaly_feature_mean': explanation.anomaly_feature_mean,
            'anomaly_feature_std': explanation.anomaly_feature_std,
            'log_bayes_factor': explanation.log_bayes_factor,
            'predicate': self.search.predicate_to_dict(explanation.predicate, include_adjacent, predicates_find_adjacent)
        }
        return res

    def explanations_to_dict(self, explanations):
        return [self.explanation_to_dict(explanation) for explanation in explanations]

    def to_dict(self):
        res = {
            'anomaly_feature': self.anomaly_feature,
            'explanation_features': self.explanation_features,
            'frontier': self.search.predicates_to_dict(self.search.frontier, self.search.base_predicates),
            'accepted': self.search.predicates_to_dict(self.search.accepted),
            'rejected': self.search.predicates_to_dict(self.search.rejected),
            'conditionally_accepted': self.search.predicates_to_dict(self.search.conditionally_accepted),
            'base_predicates': self.search.predicates_to_dict(self.search.base_predicates),
            'find_conditional': self.search.find_conditional
        }
        return res

    def load_predicate_dict(self, d, candidate_adjacent=None):
        predicate = Conjunction(d['column_to_values'], dtypes=self.data_obj.dtypes, data=self.data_obj.data, score=d['score'])
        if candidate_adjacent is not None:
            for column, adjacent_predicate_indices in d['adjacent_base_indices'].items():
                for adjacent_index in adjacent_predicate_indices:
                    predicate.set_adjacent(column, candidate_adjacent[adjacent_index])
        return predicate

    def load_predicates_dict(self, d, candidate_adjacent=None):
        return [self.load_predicate_dict(di, candidate_adjacent) for di in d]

    # def load_explanation_dict(self, d, anomaly_feature):
    #     explanation = AnomalyExplanation(anomaly_feature, self.data_obj.original_data, self.data_obj.original_dtypes, column_to_values=d['predicate']['column_to_values'],
    #                                      predicate_data=self.data_obj.data, predicate_dtypes=self.data_obj.dtypes)
    #     if 'adjacent_base_indices' in d['predicate']:
    #         for column, adjacent_predicate_indices in d['predicate']['adjacent_base_indices'].items():
    #             for adjacent_index in adjacent_predicate_indices:
    #                 explanation.predicate.set_adjacent(column, self.search.base_predicates[adjacent_index])
    #     return explanation

    # def load_explanations_dict(self, d, anomaly_feature):
    #     return [self.load_explanation_dict(di, anomaly_feature) for di in d]

    def load_dict(self, d):
        self.reset()
        anomaly_feature = d['anomaly_feature']
        base_predicates = self.load_predicates_dict(d['base_predicates'])
        
        #decide if set_features should take these as predicates or exlanations
        frontier = self.load_predicates_dict(d['frontier'], base_predicates)
        accepted = self.load_predicates_dict(d['accepted'])
        rejected = self.load_predicates_dict(d['rejected'])

        self.set_features(anomaly_feature, d['explanation_features'], base_predicates, frontier, accepted, rejected)
        self.conditionally_accepted = self.predicates_to_explanations(self.load_predicates_dict(d['conditionally_accepted']))
        self.search.find_conditional = d['find_conditional']


    # def __init__(self, df=None, dtypes=None, num_bins=15, num_points_per_bin=None, data=None, name=None, new_obj=False):
    #     self.name = name
    #     if self.name is not None and not new_obj and os.path.isfile(f'{name}.pkl'):
    #         self.load()
    #     else:
    #         if data is None:
    #             self.data = Tabular(num_bins=num_bins, num_points_per_bin=num_points_per_bin)
    #             self.data.extract(df, dtypes)
    #         else:
    #             self.data = data
    #             self.data.extract()
    #         self.search = None
    #         self.frontier = []
    #         self.accepted = []
    #         self.rejected = []
    #         self.conditionally_accepted = []
    #         self.score_f = None
    #         self.explanation_features = None
    #         self.anomaly_feature = None
    #         self.anomaly_feature_dtype = None
    #         if self.name is not None:
    #             self.save()
    #     self.bf = BayesFactor(self.data.dtypes)

    # def save(self):
    #     with open(f'{self.name}.pkl', 'wb') as f:
    #         print(self.__dict__)
    #         # pickle.dump(self.__dict__, f)

    # def load(self):
    #     with open(f'{self.name}.pkl', 'rb') as f:
    #         pxl_dict = pickle.load(f)
    #     self.__dict__.update(pxl_dict)

    # def set_features(self, anomaly_feature=None, explanation_features=None, base_predicates=None, frontier=None, accepted=None, rejected=None):
    #     """Set the anomaly feature and/or explanation features.

    #     :param anomaly_feature: Name of the column to be used as the anomaly score
    #     :type anomaly_feature: str
    #     :param explanation_features: List of columns to be included in explanations
    #     :type explanation_features: list
    #     """

    #     if anomaly_feature is not None:
    #         self.score_f = lambda mask: self.bf.bayes_factor(self.data.data, anomaly_feature, mask=mask, side='left')
    #         self.anomaly_feature = anomaly_feature
    #     if explanation_features is not None:
    #         if base_predicates is None:
    #             base_predicates = Conjunction.bottom_up_init(data_obj=self.data, columns=explanation_features)
    #         if frontier is not None:
    #             frontier = None
    #         if accepted is not None:
    #             accepted = None
    #         if rejected is not None:
    #             rejected = None
    #         self.explanation_features = explanation_features
    #     elif self.search is not None:        
    #         base_predicates = self.search.base_predicates
    #         frontier = self.search.frontier
    #         accepted = self.search.accepted
    #         rejected = self.search.rejected
    #         self.explanation_features = list(set(self.explanation_features + explanation_features))

    #     if self.anomaly_feature is not None:
    #         self.anomaly_feature_dtype = self.data.dtypes[self.anomaly_feature]
    #     if base_predicates is not None:
    #         self.search = BottomUp(self.data.data, base_predicates, self.score_f, frontier, accepted, rejected)
    #         self.frontier = self.predicates_to_explanations(self.search.frontier)
    #     if self.name is not None:
    #         self.save()

    # def predicate_to_explanation(self, predicate):
    #     """Convert the given predicate instance to an explanation instance.

    #     :param predicate: Predicate to be converted
    #     :type predicate: Predicate
    #     :return: Converted explanation
    #     :rtype: AnomalyExplanation
    #     """

    #     return AnomalyExplanation(self.anomaly_feature, self.data.original_data, self.data.original_dtypes, predicate)

    # def predicates_to_explanations(self, predicates):
    #     """Convert the multiple predicate instances to explanation instances.

    #     :param predicates: Predicates to be converted
    #     :type predicates: list
    #     :return: Converted explanations
    #     :rtype: list
    #     """

    #     explanations = sorted([self.predicate_to_explanation(predicate) for predicate in predicates], key=lambda x: x.log_bayes_factor, reverse=True)
    #     for rank_i in range(len(explanations)):
    #         adjacent_ranks = {}
    #         for feature in explanations[rank_i].explanation_features:
    #             for rank_j in range(len(explanations)):
    #                 if explanations[rank_i].predicate.is_adjacent(feature, explanations[rank_j].predicate):
    #                     if feature in adjacent_ranks:
    #                         adjacent_ranks[feature].append(rank_j)
    #                     else:
    #                         adjacent_ranks[feature] = [rank_j]
    #         explanations[rank_i].rank = rank_i
    #         explanations[rank_i].adjacent_ranks = adjacent_ranks
    #     return explanations

    def explain(self, method='expand_refine', predicates=None, maxiters=None, threshold=None, path=None, verbose=False, tracked_predicates=None):
        """Generate explanations.

        :param method: Method that will be used to generate explanations (expand, refine, or expand_refine)
        :type method: str
        :param predicates: List of predicates that will be used to start search, starts from scratch if None
        :type predicates: list
        :param threshold: If a threshold is given the search will continue until an explanation is found with a Bayes factor that exceeds that threshold
        :type threshold: float
        :return: List of explanations
        :rtype: list
        """

        if path is not None:
            subprocess.call(["python", "pixal.py", "arg1", "arg2", "argN"])

        if method == 'expand':
            predicates = self.search.expand(predicates, maxiters, 0, threshold, path, verbose, tracked_predicates)
        elif method == 'refine':
            predicates = self.search.refine(predicates, maxiters, 0, threshold, path, verbose, tracked_predicates)
        elif method == 'expand_refine':
            predicates = self.search.expand_refine(predicates, maxiters, 0, threshold, path, verbose, tracked_predicates)
        self.update_frontier_accepted_rejected()
        return self.predicates_to_explanations(predicates)
        # if predicates is None:
        #     self.conditionally_accepted = predicates
        # else:
        #     self.conditionally_accepted = self.predicates_to_explanations(predicates)
        # return self.conditionally_accepted
        # self.explanations = self.predicates_to_explanations(predicates)
        # return self.explanations

    def reset(self):
        """Reset the frontier, accepted, rejected, and conditionally accepted to empty lists.
        """

        self.frontier = []
        self.accepted = []
        self.rejected = []
        self.conditionally_accepted = []

    def get_features_explanations(self, features, explanations=None, include_subsets=True):
        """Get explanations that have been generated that include the given features.

        :param features: List of features to be included in retrieved explanations
        :type features: list
        :param explanations: List of explanations to search, searches self.explanations if None
        :type explanations: list
        :param include_subsets: Whether or not to include explanations that have a subset of the given features
        :type include_subsets: True
        """

        if explanations is None:
            explanations = self.explanations
        if include_subsets:
            features_explanations = [explanation for explanation in explanations if set(explanation.explanation_features).issubset(features)]
        else:
            features_explanations = [explanation for explanation in explanations if sorted(explanation.explanation_features) == sorted(features)]
        return features_explanations

    def plot_1d_binary_anomaly(self, explanation_feature, ax, num_bins=None):
        """Plot data with one explanation feature aßnd a binary anomaly feature.

        :param explanation_feature: Feature that will be plotted on the x-axis
        :type explanation_feature: str
        :param ax: Axis to plot on
        :type ax: matplotlib.axes
        :param num_bins: Number of bins for x-axis, defaults to self.num_bins if None
        :type num_bins: int
        """

        if num_bins is None:
            num_bins = self.num_binßs
        d = self.data_obj.original_data.groupby(
            pd.cut(self.data_obj.original_data[explanation_feature], bins=num_bins)
        )[self.anomaly_feature].agg(['mean', 'std'])
        d.index = d.index.map(lambda x: (x.left + x.right)/2)
        d.columns = [f'P({self.anomaly_feature})', 'std']
        d = d.reset_index()

        sns.lineplot(data=d, x=explanation_feature, y=f'P({self.anomaly_feature})', ax=ax)
        lower_bound = d[f'P({self.anomaly_feature})'] - d['std']
        upper_bound = d[f'P({self.anomaly_feature})'] + d['std']
        ax.fill_between(d[explanation_feature], lower_bound, upper_bound, alpha=.3)

    def plot_1d_binary_explanation(self, explanation, axis, ax):
        """Plot explanation with one explanation feature and a binary anomaly feature.

        :param explanation: Explanation to plot
        :type explanation: AnomalyExplanation
        :param axis: Axis to plot binary on, either 'x' or 'y'
        :type axis: str
        :param ax: Axis to plot on
        :type ax: matplotlib.axes
        """

        explanation_feature = explanation.explanation_features[0]
        min_val, max_val = explanation.explanation[explanation_feature]
        alpha = explanation.anomaly_feature_mean/2
        if axis == 'x':
            ax.axvspan(min_val, max_val, color='red', alpha=alpha)
        elif axis == 'y':
            ax.axhspan(min_val, max_val, color='red', alpha=alpha)

    def plot_2d_binary_explanation(self, explanation, x, y, ax):
        """Plot explanation with two explanation features and a binary anomaly feature.

        :param explanation: Explanation to plot
        :type explanation: AnomalyExplanation
        :param x: Explanation feature to plot on x-axis
        :type x: str
        :param y: Explanation feature to plot on y-axis
        :type y: str
        :param ax: Axis to plot on
        :type ax: matplotlib.axes
        """

        xy = (explanation.explanation[x][0], explanation.explanation[y][0])
        w = explanation.explanation[x][1] - explanation.explanation[x][0]
        h = explanation.explanation[y][1] - explanation.explanation[y][0]
        alpha = explanation.anomaly_feature_mean/2
        rectangle = patches.Rectangle(xy, w, h, color='red', alpha=alpha)
        ax.add_patch(rectangle)

    def plot_1d_binary(self, explanation_feature, explanations=None, ax=None, num_bins=None, verbose=False):
        """Plot data and explanations for one feature and a binary anomaly feature.

        :param explanation_feature: Feature that will be plotted on the x-axis
        :type explanation_feature: str
        :param num_bins: Number of bins for x-axis, defaults to self.num_bins if None
        :type num_bins: int
        :param ax: Axis to plot on
        :type ax: matplotlib.axes
        """

        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False
        self.plot_1d_binary_anomaly(explanation_feature, ax, num_bins)
        explanations = self.get_features_explanations((explanation_feature,), explanations)
        for explanation in explanations:
            if verbose:
                explanation.pprint()
                print()
            self.plot_1d_binary_explanation(explanation, 'x', ax)
        if return_fig:
            return fig

    def plot_2d_binary(self, x, y, explanations=None, include_subsets=True, ax=None, verbose=False):
        """Plot data and explanations for two features and a binary anomaly feature.

        :param x: Feature to plot on x-axis
        :type x: str
        :param y: Feature to plot on y-axis
        :type y: str
        :param ax: Axis to plot on
        :type ax: matplotlib.axes
        :param plot_single: Plot explanations with x and y features if True, only plot explanations with both x and y if False
        :type plot_single: bool
        """

        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        sns.scatterplot(data=self.data_obj.original_data, x=x, y=y, style=self.anomaly_feature, hue=self.anomaly_feature, ax=ax)
        explanations = self.get_features_explanations((x, y), explanations, include_subsets)
        for explanation in explanations:
            if verbose:
                explanation.pprint()
                print()
            if len(explanation.explanation_features) > 1:
                self.plot_2d_binary_explanation(explanation, x, y, ax)
            elif explanation.explanation_features[0] == x:
                self.plot_1d_binary_explanation(explanation, 'x', ax)
            elif explanation.explanation_features[0] == y:
                self.plot_1d_binary_explanation(explanation, 'y', ax)
        if return_fig:
            return fig

    def plot_feature(self, feature, explanations=None, include_subsets=True, ax=None, num_bins=None, verbose=False):
        """Plot explanations containing a given set of features along with the original data.

        :param feature: Feature or list of features to plot
        :type feature: str, list
        :param num_bins: Number of bins for x-axis, defaults to self.num_bins if None
        :type num_bins: int
        :param ax: Axis to plot on
        :type ax: matplotlib.axes
        :param plot_single: Plot explanations with x and y features if True, only plot explanations with both x and y if False
        :type plot_single: bool
        """

        if type(feature) == list:
            if self.anomaly_feature_dtype == 'binary':
                self.plot_2d_binary(feature[0], feature[1], explanations, include_subsets, ax, verbose)
        else:
            if self.anomaly_feature_dtype == 'binary':
                self.plot_1d_binary(feature, explanations, ax, num_bins, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("-m", "--method", default="expand_refine")
    parser.add_argument("-t", "--threshold", default=None)
    parser.add_argument("-i", "--maxiters", default=None)
    args = parser.parse_args()
    name = args.name
    method = args.method
    threshold = args.threshold
    maxiters = args.maxiters

    pxl = Pixal(name=name)
    explanations = pxl.explain(method, None, maxiters, threshold)
    pxl.save()
    for explanation in explanations:
        explanation.pprint()
