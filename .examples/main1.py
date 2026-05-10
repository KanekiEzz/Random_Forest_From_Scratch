import csv
import math
import random
from collections import Counter
from pathlib import Path


class _Tree_Node:
	def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value


class Calculate:
	def _entropy(self, y):
		if not y:
			return 0.0

		counts = Counter(y)
		total = len(y)
		entropy = 0.0

		for count in counts.values():
			p = count / total
			if p > 0:
				entropy -= p * math.log2(p)

		return entropy

	def H_parent(self, y):
		return self._entropy(y)

	def H_left(self, y_left):
		return self._entropy(y_left)

	def H_right(self, y_right):
		return self._entropy(y_right)

	def H_after(self, y_left, y_right):
		n = len(y_left) + len(y_right)
		if n == 0:
			return 0.0

		return (len(y_left) / n) * self._entropy(y_left) + (len(y_right) / n) * self._entropy(y_right)

	def _Information_Gain(self, y_parent, y_left, y_right):
		return self._entropy(y_parent) - self.H_after(y_left, y_right)

	def bootstrap_sample(self, X, y):
		n_samples = len(X)
		idxs = [random.randrange(n_samples) for _ in range(n_samples)]
		oob_idxs = [i for i in range(n_samples) if i not in set(idxs)]

		X_sample = [X[i] for i in idxs]
		y_sample = [y[i] for i in idxs]
		X_oob = [X[i] for i in oob_idxs]
		y_oob = [y[i] for i in oob_idxs]

		return X_sample, y_sample, X_oob, y_oob

	def _Majority_class(self, y):
		return Counter(y).most_common(1)[0][0]


class _Decision_Tree:
	def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
		self.root = None
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_features = n_features

	def fit(self, X, y):
		self.root = self._grow_tree(X, y, depth=0)

	def predict(self, X):
		return [self._traverse(x, self.root) for x in X]

	def _grow_tree(self, X, y, depth=0):
		num_samples = len(X)
		num_features = len(X[0]) if X else 0
		num_labels = len(set(y))

		if (
			depth >= self.max_depth
			or num_labels == 1
			or num_samples < self.min_samples_split
		):
			return _Tree_Node(value=Calculate()._Majority_class(y))

		if self.n_features is None or self.n_features >= num_features:
			feat_idxs = list(range(num_features))
		else:
			feat_idxs = random.sample(range(num_features), self.n_features)

		best_feature, best_threshold = self._Find_Best_Split(X, y, feat_idxs)
		if best_feature is None:
			return _Tree_Node(value=Calculate()._Majority_class(y))

		left_X, left_y, right_X, right_y = self._split(X, y, best_feature, best_threshold)
		left = self._grow_tree(left_X, left_y, depth + 1)
		right = self._grow_tree(right_X, right_y, depth + 1)
		return _Tree_Node(best_feature, best_threshold, left, right)

	def _Calculate_Entropy(self):
		print("calculate grow tree")

	def _traverse(self, x, node):
		if node.value is not None:
			return node.value
		if x[node.feature] <= node.threshold:
			return self._traverse(x, node.left)
		return self._traverse(x, node.right)

	def _split(self, X, y, feature, threshold):
		left_X, left_y, right_X, right_y = [], [], [], []
		for row, label in zip(X, y):
			if row[feature] <= threshold:
				left_X.append(row)
				left_y.append(label)
			else:
				right_X.append(row)
				right_y.append(label)
		return left_X, left_y, right_X, right_y

	def _Find_Best_Split(self, X, y, feat_idxs):
		best_gain = -1.0
		split_idx, split_threshold = None, None
		calc = Calculate()

		for feat_idx in feat_idxs:
			values = sorted(set(row[feat_idx] for row in X))
			for threshold in values:
				left_y = [label for row, label in zip(X, y) if row[feat_idx] <= threshold]
				right_y = [label for row, label in zip(X, y) if row[feat_idx] > threshold]
				if not left_y or not right_y:
					continue

				gain = calc._Information_Gain(y, left_y, right_y)
				if gain > best_gain:
					best_gain = gain
					split_idx = feat_idx
					split_threshold = threshold

		return split_idx, split_threshold


class RandomForest:
	def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
		self.n_trees = n_trees
		self.trees = []
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.n_features = n_features
		self.call = Calculate()

	def fit(self, X, y):
		self.trees = []
		if self.n_features is None:
			self.n_features = max(1, int(math.sqrt(len(X[0]))))

		for _ in range(self.n_trees):
			X_sample, y_sample, _, _ = self._bootstrap_sample(X, y)
			tree = _Decision_Tree(
				min_samples_split=self.min_samples_split,
				max_depth=self.max_depth,
				n_features=self.n_features,
			)
			tree.fit(X_sample, y_sample)
			self.trees.append(tree)

	def _bootstrap_sample(self, X, y):
		return self.call.bootstrap_sample(X, y)

	def predict(self, X):
		tree_preds = [tree.predict(X) for tree in self.trees]
		final_preds = []

		for i in range(len(X)):
			preds = [preds_for_tree[i] for preds_for_tree in tree_preds]
			final_preds.append(self._majority_vote(preds))

		return final_preds

	def _majority_vote(self, predictions):
		return Counter(predictions).most_common(1)[0][0]


def _read_csv(path):
	with open(path, newline="", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def _encode_titanic_rows(rows, is_train=True):
	sex_map = {"male": 0, "female": 1}
	embarked_map = {"S": 0, "C": 1, "Q": 2}

	ages_raw = [float(r["Age"]) for r in rows if r.get("Age") not in ("", None)]
	fares_raw = [float(r["Fare"]) for r in rows if r.get("Fare") not in ("", None)]
	age_fill = sorted(ages_raw)[len(ages_raw) // 2] if ages_raw else 0.0
	fare_fill = sorted(fares_raw)[len(fares_raw) // 2] if fares_raw else 0.0

	embarked_fill = "S"
	for r in rows:
		if r.get("Embarked"):
			embarked_fill = r["Embarked"]
			break

	X = []
	y = []

	for r in rows:
		row = [
			int(r["Pclass"]),
			sex_map.get(r["Sex"], 0),
			float(r["Age"]) if r.get("Age") not in ("", None) else age_fill,
			int(r["SibSp"]),
			int(r["Parch"]),
			float(r["Fare"]) if r.get("Fare") not in ("", None) else fare_fill,
			embarked_map.get(r.get("Embarked", embarked_fill), embarked_map[embarked_fill]),
		]
		X.append(row)

		if is_train:
			y.append(int(r["Survived"]))

	if is_train:
		return X, y
	return X


def load_data(data_dir="Data"):
	base = Path(data_dir)
	train_rows = _read_csv(base / "train.csv")
	test_rows = _read_csv(base / "test.csv")
	X_train, y_train = _encode_titanic_rows(train_rows, is_train=True)
	X_test = _encode_titanic_rows(test_rows, is_train=False)
	return X_train, y_train, X_test


def main():
	X_train, y_train, X_test = load_data()
	model = RandomForest(n_trees=10, max_depth=8)
	model.fit(X_train, y_train)

	train_preds = model.predict(X_train)
	train_acc = sum(int(p == y) for p, y in zip(train_preds, y_train)) / len(y_train)
	test_preds = model.predict(X_test)

	print(f"train accuracy: {train_acc:.4f}")
	print(f"test predictions: {test_preds[:10]}")


if __name__ == "__main__":
	main()
