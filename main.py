import numpy as np

#       (age < 30)
#       /        \
#  (salary<3k)   Class C
#    /     \
# Class A  Class B
#
# age -----> threshold
# salary ----> feature
# Class A - \
# Class B -  ---> Value
# Class C - /
#
#
# Random forests are known as ensemble learning methods used for classification and regression
#
# wla kante tree depth dyalha wasel 10 ya3eni atkoun 2047 => 2^(10+1) -1 => 2(d+1−1)


class _Tree_Node:
	def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value


class Calculate:
	# def __init__(self):
	# 	#skip now

	def _entropy(self, y: np.array) -> float:
		if len(y) == 0:
			return 0
		
		probs = np.bincount(y) / len(y)
		probs = probs[probs > 0]
		
		return -np.sum(probs * np.log2(probs))

	# Entrpy_Parent
	def H_parent(self, y: np.array) -> float:
		return self._entropy(y)

	# Entropy left child
	def H_left(self, y_left: np.array) -> float:
		return self._entropy(y_left)

	# Entropy right child
	def H_right(self, y_right: np.array) -> float:
		return self._entropy(y_right)

	# split (weghited) before Entropy
	def H_after(self, y_left, y_right):
		n = len(y_left) + len(y_right)
		
		return (
					(len(y_left) / n) * self._entropy(y_left) +
					(len(y_right) / n) * self._entropy(y_right)
				)

	# IG = H_parent = H_after
	def _Information_Gain(self, y_parent: np.array, y_left: np.array, y_right: np.array) -> float:
		print("how much entropy after use split")
		return self._entropy(y_parent) - self.H_after(y_left, y_right)

	"""
		Creates a bootstrap sample from X and y using random sampling with replacement,
		and returns the sampled data along with out-of-bag samples.
	"""
	def bootstrap_sample(self, X: np.array, y: np.array) ->  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		n_samples = X.shape[0]
		
		
		idxs = np.random.choice(n_samples, n_samples, replace=True)

		oob_idxs = np.setdiff1d(np.arange(n_samples), idxs)

		X_sample = X[idxs]
		y_sample = y[idxs]

		X_oob = X[oob_idxs]
		y_oob = y[oob_idxs]

		return X_sample, y_sample, X_oob, y_oob


	def _Majority_class(self, y):
		return np.bincount(y).argmax()


class _Decision_Tree:
	def __init__(self) -> None:
		self.root = None

	def fit(self, X, y) -> None:
		self.root = self._grow_tree(X, y, depth=0)
		# print(f"training model...\nx:{X}, y:{y}")

	def predict(self, X) -> None:
		print("predict")

	def _grow_tree(self, X, y, depth=0) -> None:
		print("grow tree")
		pass

	def _Calculate_Entropy(self) -> None:
		print("calculate grow tree")

	def _traverse(self, x, node) -> None:
		return None

	def _Find_Best_Split(self, X, y) -> None:
		print("find best split")




class RandomForest:
	def __init__(self, n_trees=10):
		self.n_trees = n_trees
		self.trees = []
		self.call = Calculate()

	def fit(self, X, y):
		self.trees = []
		
		for _ in range(self.n_trees):
			X_sample, y_sample, _, _ = self._bootstrap_sample(X, y)
		
			tree = _Decision_Tree()
			tree.fit(X_sample, y_sample)
		
			self.trees.append(tree)
		print("training model...")
		pass

	def _bootstrap_sample(self, X, y):
		return self.call.bootstrap_sample(X, y)

	def predict(self, X):
		tree_preds = []
		
		for tree in self.trees:
			tree_preds.append(tree.predict(X))
		
		tree_preds = np.array(tree_preds)
		
		final_preds = []
		
		for i in range(X.shape[0]):
			final_preds.append(self._majority_vote(tree_preds[:, i]))
		
		return np.array(final_preds)

	def _majority_vote(self, predictions):
		return np.bincount(predictions).argmax()


if __name__ == "__main__":

    np.random.seed(42)

    # -------------------------
    # SIMPLE DATASET (toy data)
    # -------------------------
    X = np.array([
        [1, 20],
        [2, 21],
        [3, 22],
        [4, 23],
        [5, 24],
        [6, 25],
        [7, 26],
        [8, 27]
    ])

    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    # -------------------------
    # CREATE MODEL
    # -------------------------
    model = RandomForest(n_trees=5)

    # -------------------------
    # TRAIN
    # -------------------------
    print("\n🔥 Training Random Forest...")
    model.fit(X, y)

    # -------------------------
    # TEST DATA
    # -------------------------
    X_test = np.array([
        [1, 20],
        [5, 24],
        [8, 27]
    ])

    # -------------------------
    # PREDICT
    # -------------------------
    print("\n🔮 Predictions...")
    preds = model.predict(X_test)

    print("Final Predictions:", preds)

	

