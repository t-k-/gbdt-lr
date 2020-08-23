from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import export_graphviz
from graphviz import Source

import numpy as np


class GBDT_LR():
    def __init__(self, n_trees=7, lr_iters=1000):
        self.gbdt = GradientBoostingClassifier(n_estimators=n_trees, init='zero')
        self.lr = LogisticRegression(max_iter=lr_iters)

    def _leaves_encoding(self, X):
        # firstly, apply GBDT to data
        leaves = self.gbdt.apply(X) # N, n_trees, n_class
        n_class = leaves.shape[2]
        leaves = np.array([row.ravel() for row in leaves]) # N, n_trees * n_class
        return leaves, n_class

    @staticmethod
    def _get_tree_num_of_leaves(trees):
        return [np.count_nonzero(tree.tree_.children_left == -1) for tree in trees]

    @staticmethod
    def _get_node_idx_of_leaves(trees):
        return [np.arange(0, tree.tree_.node_count)[tree.tree_.children_left == -1] for tree in trees]

    @staticmethod
    def _get_node2leaf_idx_map(trees):
        idx_leaves = GBDT_LR()._get_node_idx_of_leaves(trees)
        return [
            {k: v for v, k in enumerate(nodes)}
            for nodes in idx_leaves
        ]


    def _onehot_encoding(self, X):
        trees = self.gbdt.estimators_.ravel()
        num_leaves = self._get_tree_num_of_leaves(trees)
        node2leaf = self._get_node2leaf_idx_map(trees)
        X_leaves, n_class = self._leaves_encoding(X)

        onehots = []
        for leaves in X_leaves:
            leaves = map(lambda x: node2leaf[x[0]][int(x[1])], enumerate(leaves))
            row = np.array([], dtype=int)
            for tree_idx, leaf in enumerate(leaves):
                onehot = np.zeros((num_leaves[tree_idx]), dtype=int)
                onehot[leaf] = 1
                row = np.concatenate((row, onehot))
            onehots.append(row)
        return np.array(onehots), n_class

    def sum_leaf_value(self, X, n_class):
        trees = self.gbdt.estimators_.ravel()
        X_leaves, _ = self._leaves_encoding(X)
        results = []
        for leaves in X_leaves:
            sum_leaf_vals = [0] * n_class
            for i, leaf in enumerate(leaves.ravel()):
                class_idx = i % n_class
                leaf_val = trees[i].tree_.value[int(leaf)][0][0]
                sum_leaf_vals[class_idx] += leaf_val
            results.append(sum_leaf_vals)
        return results

    def tree_predict(self, X):
        return self.gbdt.predict(X)

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        trees = self.gbdt.estimators_.ravel()
        onehots, n_class = self._onehot_encoding(X)
        self.lr.fit(onehots, y)
        return trees, n_class

    def predict(self, X):
        onehots, _ = self._onehot_encoding(X)
        pred_probs = self.lr.predict_proba(onehots)
        pred_class = self.lr.predict(onehots)
        return pred_class, pred_probs

    @staticmethod
    def render_trees(trees):
        for i, tree in enumerate(trees):
            dot = export_graphviz(tree, out_file=None, node_ids=True)
            Source(dot, format="png").render(filename=f"./tree-{i}")
        print('Done.')


if __name__ == '__main__':
    debug = False

    # generate some data
    X, y = make_classification(n_samples=800, n_classes=4, n_informative=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = GBDT_LR()
    trees, n_class = model.fit(X_train, y_train)
    print('[classes]', n_class)
    if debug:
        model.render_trees(trees)
    preds, probs = model.predict(X_test)

    tree_preds = model.tree_predict(X_test)
    if debug:
        sums = model.sum_leaf_value(X_test, n_class)
        sum_preds = [np.argmax(s) for s in sums]
        print(any(tree_preds != sum_preds))

    print('GBDT test accuracy:', np.count_nonzero(tree_preds == y_test) / len(X_test))
    print('GBDT+LR test accuracy:', np.count_nonzero(preds == y_test) / len(X_test))
