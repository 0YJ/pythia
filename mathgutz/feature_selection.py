from collections import defaultdict
import numpy
from scipy import stats

# Compute the information gain from splitting over feature f, coming from
# a 2x2 contingency table obs with total n = a+b+c+d data points
#
# obs | c=1 | c=0
# f=1 | a   | b
# f=0 | c   | d

def plogp(x):
    if x > 0:
        return x * numpy.log2(x)
    else:
        return 0.0

def info_gain(obs):
    a, b = float(obs[0][0]), float(obs[0][1])
    c, d = float(obs[1][0]), float(obs[1][1])
    n = max(a + b + c + d, 1)
    h_before = -1 * (plogp((a+c)/n) + plogp((b+d)/n))
    h_left = -1 * (plogp(a/(a+b)) + plogp(b/(a+b)))
    h_right = -1 * (plogp(c/(c+d)) + plogp(d/(c+d)))
    h_after = ((a+b)/n) * h_left + ((c+d)/n) * h_right
    return h_before - h_after

class FeatureSelector:
    # Pass in a training hash of form {id: {"features":{feature dictionary}, "label":n} }
    # where n = 0 (false) or n = 1 (true)
    def __init__(self, train_hash):
        self.best = []
        self.train_set = train_hash.keys()
        self.train_weights = dict([(id, train_hash[id]["features"]) for id in self.train_set])
        self.features = dict([(id, train_hash[id]["features"].keys()) for id in self.train_set])
        self.label_func = dict([(id, train_hash[id]["label"]) for id in self.train_set])
        self.feature_rank = defaultdict(lambda: defaultdict(float))
        self.feature_set = set([])
        self.metrics_ranked = set([])
        for id in self.train_set:
            for feat in self.features[id]:
                self.feature_rank["frequency"][feat] += 1
                self.feature_set.add(feat)
    def rank_features(self, metric):
        self.metrics_ranked.add(metric)
        for feat in self.feature_set:
            feat_func = {}
            for id in self.train_set:
                if feat in self.features[id]:
                    feat_func[id] = 1
                else:
                    feat_func[id] = 0
            if metric == "info":
                feat_yes = set([id for id in self.train_set if feat_func[id] == 1])
                feat_no = set([id for id in self.train_set if feat_func[id] == 0])
                label_yes = set([id for id in self.train_set if self.label_func[id] == 1])
                label_no = set([id for id in self.train_set if self.label_func[id] == 0])
                x = [len(feat_yes & label_yes), len(feat_yes & label_no)]
                y = [len(feat_no & label_yes), len(feat_no & label_no)]
                a, b, c, d = x[0], x[1], y[0], y[1]
                obs = numpy.array([x, y])
                self.feature_rank["info"][feat] = info_gain(obs)
            elif metric == "spearman":
                u = [self.label_func[id] for id in self.train_set]
                v = [feat_func[id] for id in self.train_set]
                rho, pval = stats.spearmanr(u, v)
                self.feature_rank["spearman"][feat] = abs(rho)
            else:
                feat_yes = set([id for id in self.train_set if feat_func[id] == 1])
                feat_no = set([id for id in self.train_set if feat_func[id] == 0])
                label_yes = set([id for id in self.train_set if self.label_func[id] == 1])
                label_no = set([id for id in self.train_set if self.label_func[id] == 0])
                x = [len(feat_yes & label_yes), len(feat_yes & label_no)]
                y = [len(feat_no & label_yes), len(feat_no & label_no)]
                a, b, c, d = x[0], x[1], y[0], y[1]
                obs = numpy.array([x, y])
                chi2, pval, dof, ex = stats.chi2_contingency(obs, correction=False)
                self.feature_rank[metric][feat] = 1-pval
    def return_features(self, metric):
        # Get a list of all training features ranked by metric in descending order
        if not metric in self.metrics_ranked:
            self.rank_features(metric)
        return sorted([(feat, self.feature_rank[metric][feat]) for feat in self.feature_set], key=lambda x: x[1], reverse=True)
    def select_k_best(self, k, metric):
        # Add the top k features to the list of important features, ranked by metric
        if not metric in self.metrics_ranked:
            self.rank_features(metric)
        newhash = self.feature_rank[metric]
        feats = sorted(newhash.keys(), key=lambda x: newhash[x], reverse=True)
        self.best = feats[:k]
    def select_k_perc(self, k, metric):
        # Add the top k percent of features to the list of important features, ranked by metric
        if not metric in self.metrics_ranked:
            self.rank_features(metric)
        newhash = self.feature_rank[metric]
        feats = sorted(newhash.keys(), key=lambda x: newhash[x], reverse=True)
        goal = int(float(k * len(feats)) / 100.0)
        self.best = feats[:goal]
    def select_threshold(self, thresh, metric):
        # Add all features with metric value above thresh to the list of important features
        if not metric in self.metrics_ranked:
            self.rank_features(metric)
        newhash = self.feature_rank[metric]
        self.best = [x for x in self.feature_set if newhash[x] > thresh]
    def return_best(self, metric):
        # Return the current list of important features, ranked by the appropriate metric
        return sorted([(feat, self.feature_rank[metric][feat]) for feat in self.best], key=lambda x: x[1], reverse=True)
    def add_to_best(self, more_words):
        # Add user-defined list of words to the list of important features
        self.best = list(set(self.best) | (set(self.feature_set) & set(more_words)))
    def training_features(self):
        # The output of this can be used in an NLTK classifier's "train" method
        newfeatures = {}
        for id in self.train_set:
            oldfeats = set(self.features[id])
            newfeats = dict([(feat, self.train_weights[id][feat]) for feat in (oldfeats & set(self.best))])
            newfeatures[id] = newfeats
        return [(newfeatures[id], self.label_func[id]) for id in self.train_set]
    def test_features(self, test_hash):
        # test_hash should be a dict of form {id: "features":{feature dict}, "label":n }
        # result[id] is suitable for piping into an NLTK classifier
        test_set = test_hash.keys()
        test_weights = dict([(id, test_hash[id]["features"]) for id in test_set])
        result = {}
        for id in test_set:
            oldfeats = set(test_hash[id]["features"].keys())
            newfeats = dict([(feat, test_weights[id][feat]) for feat in (oldfeats & set(self.best))])
            result[id] = (newfeats, test_hash[id]["label"])
        return result
    def select_features(self, real_corpus):
        # test_corpus should be a dict of form {id: {feature dict} }
        # result[id] is suitable for piping into an NLTK classifier
        real_set = real_corpus.keys()
        result = {}
        for id in real_set:
            oldfeats = set(real_corpus[id].keys())
            newfeats = dict([(feat, real_corpus[id][feat]) for feat in (oldfeats & set(self.best))])
            result[id] = newfeats
        return result
