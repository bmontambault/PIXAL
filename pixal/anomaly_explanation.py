import numpy as np
import json
from sklearn.cluster import SpectralClustering

from bayes_factor.bayes_factor import BayesFactorOld
from predicate_induction.predicate import Conjunction
from predicate_induction.utils import plot_predicate_dist, plot_predicate_pivot

class AnomalyExplanation:
    
    def __init__(self, data, dtypes, feature_values, feature_mask, score_col):
        self.data = data
        self.dtypes = dtypes
        self.feature_values = feature_values
        self.feature_mask = feature_mask
        self.score_col = score_col
        
        self.mask = self.feature_mask.all(axis=1)
        self.support = self.mask.sum()
        self.bf = BayesFactorOld({self.score_col: 'numeric'})
        self.log_bayes_factor = self.get_log_bayes_factor()
        
    def get_log_bayes_factor(self):
        return self.bf.bayes_factor(self.data, self.score_col, mask=self.mask)
        
    def plot(self, x=None, y=None, bins=None, ax=None):
        if y is None:
            y = self.score_col
        if x is None:
            return plot_predicate_dist(self.data, self.feature_mask, y, ax=ax)
        else:
            return plot_predicate_pivot(self.data, self.dtypes, self.feature_mask, x, y, bins=bins, ax=ax);        
        
    def __repr__(self):
        return str(self.feature_values)

class AnomalyExplanations:
    
    def __init__(self, data, dtypes, score_col, explanations):
        self.data = data
        self.dtypes = dtypes
        self.score_col = score_col
        self.explanations = explanations
        
        self.mask = np.array([p.mask.values for p in self.explanations])
        self.coverage = self.mask[:,None] * self.mask[None,:]
        self.support = self.mask.sum(axis=1)
        self.similarity = self.coverage.sum(axis=2) / self.support
        self.score = (self.data[self.score_col].values[None,:] * self.mask).sum(axis=1) / self.support
        self.log_bayes_factor = [e.log_bayes_factor for e in self.explanations]
        self.clusters = self.get_clusters(self.similarity)
        
        # self.nodes = self.get_nodes()
        # self.links = self.get_links(self.nodes)

    def score_label(self, similarity, labels, label):
        similarity_ = similarity[labels==label]
        in_sim = similarity_[:,labels==label].mean()
        out_sim = similarity_[:,labels!=label].mean()
        return in_sim * (1-out_sim)

    def score_labels(self, similarity, labels):
        return [self.score_label(similarity, labels, label) for label in labels]

    def get_clusters(self, similarity):
        all_scores = []
        all_labels = []
        for n_clusters in range(2, similarity.shape[0]+1):
            labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(similarity).labels_
            scores = self.score_labels(similarity, labels)
            all_scores.append(scores)
            all_labels.append(labels)
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        return all_labels[all_scores.mean(axis=1).argmax()]
        
    def get_nodes(self):
        nodes = [{'id': i, 'name': i, 'support': self.support[i].item(), self.score_col: self.score[i].item(),
                 'evidence': self.log_bayes_factor[i].item()} for i in range(len(self.explanations))]
        
        groups = {}
        new_group = 0
        for node in nodes:
            connected_nodes = [g for g in groups.keys() if self.similarity[node['id']][g] > 0]
            if len(connected_nodes) == 0:
                new_group += 1
                groups[node['id']] = new_group
            else:
                group = groups[connected_nodes[0]]
                for i in connected_nodes:
                    groups[i] = group
                groups[node['id']] = group
        keys = list(set(groups.values()))
        for i in range(len(keys)):
            for k, v in groups.items():
                if v == keys[i]:
                    groups[k] = i
        for k, v in groups.items():
            nodes[k]['group'] = v
        return nodes
    
    def get_links(self, nodes):
        links = []
        for i in range(len(self.similarity)):
            for j in range(len(self.similarity[i])):
                if i < j:
                    links.append({'source': i, 'target': j, 'similarity': self.similarity[i][j]})

        for l in links:
            source_group = nodes[l['source']]['group']
            target_group = nodes[l['target']]['group']
            if source_group == target_group and l['similarity'] > 0:
                l['group'] = source_group
            else:
                l['group'] = -1
        return links
    
    def to_json(self, name):
        with open(f'{name}_nodes_links.json', 'w') as f:
            json.dump({'nodes': self.nodes, 'links': self.links}, f)
        with open(f'{name}_predicates.json', 'w') as f:
            json.dump({i: self.explanations[i].feature_values for i in range(len(self.explanations))}, f)
        with open(f'{name}_predicate_masks.json', 'w') as f:
            json.dump({i: self.explanations[i].feature_mask.to_dict() for i in range(len(self.explanations))}, f)


class AnomalyExplanationOld:
    """This class wraps a predicate and adds additional information for plotting.

    :param anomaly_feature: Name of the column used as the anomaly score
    :type anomaly_feature: str
    :param data: Data that this explanation refers to
    :type data: pd.DataFrame
    :param predicate: Predicate object defining the points covered by this explanation
    :type predicate: Predicate
    """

    def __init__(self, anomaly_feature, data, dtypes, predicate=None, column_to_values=None, predicate_data=None, predicate_dtypes=None, anomaly_feature_mean=None, anomaly_feature_std=None, log_bayes_factor=None, adjacent_ranks=None):
        if predicate is None:
            self.predicate = Conjunction(column_to_values, predicate_dtypes, predicate_data)
        else:
            self.predicate = predicate
        self.explanation_features = self.predicate.columns
        self.explanation = {}
        for col in self.predicate.columns:
            if dtypes[col] in ['numeric', 'ordinal']:
                min_val = data.loc[self.predicate.mask][col].min()
                max_val = data.loc[self.predicate.mask][col].max()
                self.explanation[col] = [min_val, max_val]
            elif dtypes[col] in ['nominal', 'binary']:
                self.explanation[col] = self.predicate.column_to_values[col]
        self.anomaly_feature = anomaly_feature
        if anomaly_feature_mean is None:
            self.anomaly_feature_mean = data[anomaly_feature].loc[self.predicate.mask].mean()
        else:
            self.anomaly_feature_mean = anomaly_feature_mean
        if anomaly_feature_std is None:
            self.anomaly_feature_std = data[anomaly_feature].loc[self.predicate.mask].std()
        else:
            self.anomaly_feature_std = anomaly_feature_std
        if log_bayes_factor is None:
            self.log_bayes_factor = self.predicate.score
        else:
            self.log_bayes_factor = log_bayes_factor
        self.count = self.predicate.mask.sum()

        self.info = {'explanation': self.explanation}
        if dtypes[anomaly_feature] == 'binary':
            self.info[f'P({anomaly_feature}|explanation)'] = self.anomaly_feature_mean
            self.info['count'] = self.count
            self.info['log[P(H1)/P(H0)]'] = self.log_bayes_factor
        self.adjacent_ranks = adjacent_ranks

    def convert_numpy(self, d):
        if type(d) == dict:
            return {k: v.item() if hasattr(v, 'item') else self.convert_numpy(v) if type(v) in [dict, list] else v for k, v in d.items()}
        elif type(d) == list:
            return [di.item() if hasattr(di, 'item') else di for di in d]

    def to_dict(self):
        explanation_json = {
            "log_bayes_factor": self.log_bayes_factor,
            "support": self.count,
            "anomaly_feature_mean": self.anomaly_feature_mean,
            "anomaly_feature_std": self.anomaly_feature_std,
            "explanation": self.explanation,
            "column_to_values": self.predicate.column_to_values,
            "adjacent_ranks": self.adjacent_ranks,
            "rank": self.rank
        }
        explanation_json = self.convert_numpy(explanation_json)
        return explanation_json

    def __eq__(self, other):
        if isinstance(other, AnomalyExplanation):
            return self.predicate == other.predicate
        return False

    def pprint_dict(self, d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print('\t' * indent + str(key))
                self.pprint_dict(value, indent+1)
            else:
                print('\t' * indent + str(key) + ': ' + str(value))

    def pprint(self):
        self.pprint_dict(self.info)

    def __repr__(self):
        return str(self.explanation)