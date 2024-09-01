import numpy as np

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._fit(X, y, depth=0)

    def _fit(self, X, y, depth):
        if len(set(y)) == 1:
            return DecisionTreeNode(value=y[0])
        if depth >= self.max_depth:
            return DecisionTreeNode(value=self._most_common_label(y))

        num_samples, num_features = X.shape
        best_split = self._best_split(X, y)
        if best_split is None:
            return DecisionTreeNode(value=self._most_common_label(y))

        left_indices = X[:, best_split['feature']] <= best_split['threshold']
        right_indices = X[:, best_split['feature']] > best_split['threshold']

        left_node = self._fit(X[left_indices], y[left_indices], depth + 1)
        right_node = self._fit(X[right_indices], y[right_indices], depth + 1)
        return DecisionTreeNode(feature=best_split['feature'],
                                threshold=best_split['threshold'],
                                left=left_node,
                                right=right_node)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float('inf')
        best_split = None
        for feature in range(num_features):
            feature_values = np.unique(X[:, feature])
            for threshold in feature_values:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gini = self._gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature': feature, 'threshold': threshold}
        return best_split

    def _gini_index(self, left_labels, right_labels):
        left_size = len(left_labels)
        right_size = len(right_labels)
        total_size = left_size + right_size
        left_gini = 1.0 - sum((np.sum(left_labels == label) / left_size) ** 2 for label in np.unique(left_labels))
        right_gini = 1.0 - sum((np.sum(right_labels == label) / right_size) ** 2 for label in np.unique(right_labels))
        return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(sample, self.root) for sample in X])

    def _predict(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict(sample, node.left)
        else:
            return self._predict(sample, node.right)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        self.trees = []
        num_features = X.shape[1]
        if self.max_features is None:
            self.max_features = num_features

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train decision tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.array([np.bincount(predictions).argmax() for predictions in tree_predictions.T])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForest(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")