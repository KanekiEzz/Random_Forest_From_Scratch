from ensurepip import bootstrap
from itertools import count
import os
from statistics import mode
from sys import modules
import numpy as np
import math
from collections import Counter
from collections.abc import Hashable, Sequence



"""
 				RandomForest
					   │
		┌──────────────┼──────────────┐
		│              │              │
		▼              ▼              ▼
	DecisionTree    DecisionTree   DecisionTree
	(_Decision_Tree)(_Decision_Tree)(_Decision_Tree)
		│              │              │
		▼              ▼              ▼
	Tree 1         Tree 2         Tree 3
"""


# DEBUG = False
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")


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
  

"""
	Base class for all machine learning models in this implementation.
	
	This class defines a common interface for training (fit) and inference (predict),
	ensuring that all derived models follow a consistent structure.
	
	It also provides a built-in logging system controlled by a debug flag,
	allowing developers to enable or disable colored debug output during development.
	
	The logging utility supports multiple levels (debug, info, warning, error),
	which helps in tracking model behavior during training and debugging complex workflows
	such as decision tree construction and ensemble learning methods like Random Forest.
"""
class BaseModel:
    RESET = "\033[0m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"

    def __init__(self, debug: bool = DEBUG):
        self.debug = debug

    def log(self, msg, level="debug"):
        if not self.debug:
            return

        if level == "debug":
            color = self.BLUE
            tag = "DEBUG"
        elif level == "info":
            color = self.GREEN
            tag = "INFO"
        elif level == "warn":
            color = self.YELLOW
            tag = "WARN"
        elif level == "error":
            color = self.RED
            tag = "ERROR"
        else:
            color = self.RESET
            tag = "LOG"

        print(f"{color}[{tag}] {msg}{self.RESET}")

    def fit(self, X, y):
        raise NotImplementedError("fit() must be implemented")

    def predict(self, X):
        raise NotImplementedError("predict() must be implemented")


"""
	convert string to integer 
"""
class LabelEncoderSimple(BaseModel):
	def __init__(self, debug=DEBUG):
		super().__init__(debug)
		self.class_to_index = {}
		self.index_to_class = {}

	def fit(self, y) -> None:
		unique_classes = sorted(set(y)) # remover dublicate and sorte
		self.log(f"HI {unique_classes}")

		if (self.debug):# ['cat': 0, 'dog': 1, 'bird', 2]
			for index, c in enumerate(unique_classes):
				self.class_to_index[c] = index
				self.log(f"inex: {index}", "info")
				self.log(f"c: {c}", "error")
		else:
			self.class_to_index = {c: index for index, c in enumerate(unique_classes)}
	
		self.log(f"class to index: {self.class_to_index}")

		if (self.debug):# [0: 'cat', 1: 'dog', 2: 'bird']
			for index, c in self.class_to_index.items():
				self.index_to_class[c] = index
				self.log(f"index to class: {self.index_to_class}", "warn")
		else:
			self.index_to_class = {i: c for c, i in self.class_to_index.items()}

	def transform(self, y) -> tuple[np.array]:
		return np.array([self.class_to_index.get(c, -1) for c in y])

	def inverse_transform(self, y_encoded) -> tuple[np.array]:
		return np.array([self.index_to_class.get(i, None) for i in y_encoded])

class Utils():
    # def __init__(self, ):
	@staticmethod
	def candidate_thresholds(values: np.ndarray, max_thresholds: int = 32) -> np.ndarray:
		"""
			Return candidate thresholds between sorted unique values.
		"""

		if max_thresholds < 1:
			raise ValueError("max_thresholds must be at least 1.")

		unique_values = np.unique(values)

		if unique_values.size < 2:
			return np.array([], dtype=float)

		# Midpoints between consecutive unique values
		thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

		if thresholds.size <= max_thresholds:
			return thresholds

		if max_thresholds == 1:
			return np.array([thresholds[thresholds.size // 2]])

		# Evenly sample thresholds
		indices = np.linspace(
			0,
			thresholds.size - 1,
			max_thresholds,
			dtype=int,
		)

		return thresholds[np.unique(indices)]

    

class Calculate(BaseModel, Utils):
	def __init__(self, debug: bool = DEBUG):
		super().__init__(debug)

	"""
	    Calculate entropy step by step.
	
	    Example:
	        y = [1, 2, 2, 4, 1]
	
	    Step 1:
	        Count how many times each class appears.
	
	        class 1 -> 2 times
	        class 2 -> 2 times
	        class 4 -> 1 time
	
	    Step 2:
	        Convert counts into probabilities.
	
	        P(1) = 2 / 5
	        P(2) = 2 / 5
	        P(4) = 1 / 5
	
	    Step 3:
	        Apply entropy formula.
	
	        H(Y) = -Σ( p * log2(p) )
	
	    Entropy measures how mixed or impure the data is.
	
	    - Entropy = 0:
	        All samples belong to one class (pure node)
	
	    - Higher entropy:
	        Classes are mixed together
	"""
	def _entropy(self, y: np.array) -> float:
		if len(y) == 0:
			return 0.0
		
		counts = np.bincount(y) # convert how much deblicate[1, 1, 2, 2] =\> [0 2 2]
		self.log(f"y: {y}", "info")
		self.log(f"counts by use bincount: {counts}")
		
		probs = counts / len(y)
		self.log(f"props: {probs}")
		
		
		probs = probs[probs > 0]
		self.log(f"probs: {probs}","error")
		
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

		if n == 0:
			return 0.0
		
		return ( (len(y_left) / n) * self._entropy(y_left) +
					(len(y_right) / n) * self._entropy(y_right)
     			)

	def gini_after(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
		n = len(y_left) + len(y_right)

		if n == 0:
			return 0.0

		return (
			(len(y_left) / n) * self.gini(y_left)
			+ (len(y_right) / n) * self.gini(y_right)
		)
  
	def gini(self, y):
		if len(y) == 0:
			return 0.0

		counts = np.bincount(y)
		probs = counts / len(y)
		probs = probs[probs > 0]

		return 1.0 - np.sum(probs ** 2)

	# IG = H_parent = H_after
	def information_gain(self, y_parent: np.array, y_left: np.array, y_right: np.array) -> float:
		return self._entropy(y_parent) - self.H_after(y_left, y_right)

	def gini_gain(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
		return self.gini(y_parent) - self.gini_after(y_left, y_right)

	"""
		Creates a bootstrap sample from X and y using random sampling with replacement,
		and returns the sampled data along with out-of-bag samples.
	"""
	def bootstrap_sample(self, X: np.array, y: np.array, rng: np.random.Generator) ->  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		n_samples = X.shape[0]
		self.log(f"n_samples: {n_samples}")
		
		
		# idxs = np.random.choice(n_samples, n_samples, replace=True)
		idxs = rng.choice(
			n_samples,
			size=n_samples,
   			replace=True
		)

		oob_idxs = np.setdiff1d(np.arange(n_samples), idxs)
		self.log(f"range: {np.arange(n_samples)}", "error")

		X_sample = X[idxs]
		y_sample = y[idxs]

		X_oob = X[oob_idxs]
		y_oob = y[oob_idxs]

		return X_sample, y_sample, X_oob, y_oob

	"""
	    Return the most frequent class label in y.
	
	    This method is used mainly in Decision Trees when creating
	    a leaf node. If the algorithm stops splitting the data,
	    the node predicts the class that appears the most.
	
	    Example:
	        y = [0, 1, 1, 1, 0]
	
	        np.bincount(y) -> [2, 3]
	
	        class 0 appears 2 times
	        class 1 appears 3 times
	
	        argmax() returns index of largest value -> 1
	
	    Result:
	        1
	"""
	def _Majority_class(self, y: np.array) -> int:
		return np.bincount(y).argmax()


class _Decision_Tree(BaseModel):
	def __init__(self,
		min_samples_split=2,
		max_depth=100,
		n_features=None,
		rng=None,
		max_thresholds=32,
		criterion="entropy",
		debug: bool = DEBUG
	) -> None:
		super().__init__(debug)
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_features = n_features
		self.root = None
		self.rng = rng or np.random.default_rng()
		self.max_thresholds = max_thresholds
		self.criterion=criterion
		self.calc = Calculate(debug=debug)

	def fit(self, X, y) -> None:
		# if n_features not set => use all features
		self.n_features = self.n_features or X.shape[1]
		self.root = self._grow_tree(X, y, depth=0)

	def predict(self, X) -> np.ndarray:
		# traverse the tree for every row in X
		return np.array([self._traverse(x, self.root) for x in X])

	def _grow_tree(self, X, y, depth=0):
	
		n_samples, n_features = X.shape # size columns(features) and rows(samples)

		n_labels = len(np.unique(y))  # because it tells us if the node is pure (only 1 class) or still needs splitting
	
		# stop conditions
		if (
			depth >= self.max_depth
			or n_labels == 1
			or n_samples < self.min_samples_split
		):
			leaf_value = self.calc._Majority_class(y)
			return _Tree_Node(value=leaf_value)
	
		# find best split
		best_feature, best_threshold = self._Find_Best_Split(X, y)
	
		# if no split found => leaf
		if best_feature is None:
			leaf_value = self.calc._Majority_class(y)
			return _Tree_Node(value=leaf_value)
	
		# split using numpy masks
		left_idxs  = X[:, best_feature] <= best_threshold
		right_idxs = X[:, best_feature] >  best_threshold
	
		X_left  = X[left_idxs]
		y_left  = y[left_idxs]
	
		X_right = X[right_idxs]
		y_right = y[right_idxs]
	
		# recursive grow
		left  = self._grow_tree(X_left,  y_left,  depth + 1)
		right = self._grow_tree(X_right, y_right, depth + 1)
	
		return _Tree_Node(
			feature=best_feature,
			threshold=best_threshold,
			left=left,
			right=right
		)

	def _Calculate_Entropy(self) -> None:
		print("calculate grow tree")

	def _traverse(self, x, node) -> int:
		"""
		Walk a single sample x down the tree until we hit a leaf.

		Example tree:
		    (age <= 30)
		    /          \
		(sal <= 3k)   Class C (value=2)
		  /     \
		Class A  Class B

		x = [25, 2000]
		  -> age=25 <= 30  => go LEFT
		  -> salary=2000 <= 3000 => go LEFT
		  -> leaf: value = Class A
		"""
		# leaf node => return the stored class label
		if node.value is not None:
			return node.value

		# go left or right depending on the threshold
		if x[node.feature] <= node.threshold:
			return self._traverse(x, node.left)
		return self._traverse(x, node.right)

	def _Find_Best_Split(self, X, y) -> tuple:
		"""
		Try every feature and every unique value as a threshold.
		Pick the (feature, threshold) with the highest Information Gain.

		Steps
		-----
		1. Pick a random subset of features (size = self.n_features).
		   This is the key Random Forest trick: not all features every split.

		2. For each feature, collect all unique values as candidate thresholds.

		3. For each threshold split the rows into left (<= threshold) and right (> threshold).

		4. Compute IG = H(parent) - weighted_H(children).

		5. Return the best (feature, threshold).

		Example
		-------
		X = [[1, 20],       feature 0: age
		     [3, 22],       feature 1: salary
		     [5, 24]]
		y = [0, 0, 1]

		Trying feature=0, threshold=3:
		  left  = rows where age <= 3 => y_left  = [0, 0]
		  right = rows where age >  3 => y_right = [1]
		  IG = H([0,0,1]) - weighted_H([0,0], [1])
		     = 0.918       - 0.0
		     = 0.918   <-- best so far => keep it
		"""
		best_gain      = -1.0
		best_feature   = None
		best_threshold = None

		# 1) random subset of feature indices
		feat_idxs = self.rng.choice(X.shape[1], self.n_features, replace=False)

		for feat in feat_idxs:
			col        = X[:, feat]
			# thresholds = np.unique(col)  # every unique value is a candidate
			thresholds = self.calc.candidate_thresholds(col, self.max_thresholds)

			for threshold in thresholds:
				left_mask  = col <= threshold
				right_mask = ~left_mask

				# skip useless splits where one side is empty
				if left_mask.sum() == 0 or right_mask.sum() == 0:
					continue
				
				if self.criterion == "entropy":
					gain = self.calc.information_gain(y, y[left_mask], y[right_mask])
				else:
					gain = self.calc.gini_gain(y, y[left_mask], y[right_mask])

				self.log(f"feat={feat} thr={threshold:.3f} IG={gain:.4f}")

				if gain > best_gain:
					best_gain      = gain
					best_feature   = feat
					best_threshold = threshold

		return best_feature, best_threshold


class RandomForest(BaseModel):
	"""
			random_state = 42
					│
					▼
			np.random.default_rng(42)
					│
					▼
				rng
				/   \
				/     \
			bootstrap   feature selection
	"""
	def __init__(self, n_trees=10, max_depth=100, min_samples_split=2, n_features=None, random_state = None, criterion="entropy", debug: bool = DEBUG):
		super().__init__(debug)
		self.n_trees           = n_trees
		self.max_depth         = max_depth
		self.min_samples_split = min_samples_split
		self.n_features        = n_features   # None => sqrt(total_features) decided inside _Decision_Tree
		self.random_state	   = random_state
		self.rng 			   = np.random.default_rng(random_state)
		self.trees             = []
		self.call              = Calculate()
		self.criterion		   = criterion
		self.encoder           = LabelEncoderSimple(debug=debug)
		self._needs_encoding   = False        # flipped to True when y contains strings
		if criterion not in ("entropy", "gini"):
			raise ValueError(
				f"Invalid criterion '{criterion}'. Expected 'entropy' or 'gini'."
			)

	def fit(self, X, y):
		X = np.array(X, dtype=float)
		y = np.array(y)

		# ── encode string labels ──────────────────────────────────────────────
		# supports both int labels (0,1,2) and string labels ("cat","dog","bird")
		if y.dtype.kind in ("U", "S", "O"):   # Unicode / bytes / object => strings
			self._needs_encoding = True
			self.encoder.fit(y)
			y = self.encoder.transform(y)
		else:
			y = y.astype(int)
		# ─────────────────────────────────────────────────────────────────────

		self.trees = []
		
		for i in range(self.n_trees):
			X_sample, y_sample, _, _ = self._bootstrap_sample(X, y)
		
			tree = _Decision_Tree(
				min_samples_split=self.min_samples_split,
				max_depth=self.max_depth,
				n_features=self.n_features or max(1, int(np.sqrt(X.shape[1]))),
				criterion=self.criterion,
    			rng = self.rng,
    			debug=self.debug, 
			)
			tree.fit(X_sample, y_sample)
		
			self.trees.append(tree)
			self.log(f"tree {i + 1}/{self.n_trees} trained", "info")

	def _bootstrap_sample(self, X: np.array, y: np.array):
		return self.call.bootstrap_sample(np.array(X), np.array(y), self.rng)

	def predict(self, X):
		X = np.array(X, dtype=float)

		# collect every tree's predictions => shape (n_trees, n_samples)
		tree_preds = np.array([tree.predict(X) for tree in self.trees])
		
		# majority vote for each sample
		final_preds = np.array([
			self._majority_vote(tree_preds[:, i])
			for i in range(X.shape[0])
		])

		# ── decode back to original string labels if needed ───────────────────
		if self._needs_encoding:
			return self.encoder.inverse_transform(final_preds)
		return final_preds

	def _majority_vote(self, predictions):
		return np.bincount(predictions).argmax()




# ── test with int labels ──────────────────────────────────────────────────
def test_with_int_labels():
	print("\n--- Test 1: int labels ---")
	X_train = np.array([
		[1, 20], [2, 21], [3, 22], [4, 23],
		[5, 24], [6, 25], [7, 26], [8, 27],
	])
	y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])

	rf = RandomForest(n_trees=5, max_depth=5, random_state=42, criterion="gini")
	rf.fit(X_train, y_train)

	X_test = np.array([[1, 20], [5, 24], [8, 27]])
	preds  = rf.predict(X_test)
	print("Predictions:", preds)   # expected: [0 1 1]
	print("Expected:   ", [0, 1, 1])


# ── test with string labels ───────────────────────────────────────────────
def test_with_string_labels():
	print("\n--- Test 2: string labels ---")
	X_train2 = np.array([
		[1, 10], [2, 11], [3, 12],
		[7, 20], [8, 21], [9, 22],
		[4, 30], [5, 31], [6, 32],
	])
	y_train2 = ["cat", "cat", "cat", "dog", "dog", "dog", "bird", "bird", "bird"]

	rf2 = RandomForest(n_trees=100, max_depth=5)
	rf2.fit(X_train2, y_train2)

	X_test2 = np.array([[2, 11], [8, 21], [5, 31]])
	preds2  = rf2.predict(X_test2)
	print("Predictions:", preds2) 
	print("Expected:   ", ["cat", "dog", "bird"])



def just_majority_class():
	# test majority Class
	y = [0, 1, 1, 1, 0]
	majority = Calculate()
	p = majority._Majority_class(y)
	model.log(f"P: {p}")
	model.log(f"this a bage value: {y[p]}", "error")

	np.random.seed(42)



def just_test_DecisionTree():
	print("\n--- Test 1: Decision_trre ---")
	X_train = np.array([
		[1, 20], [2, 21], [3, 22], [4, 23],
		[5, 24], [6, 25], [7, 26], [8, 27],
	])
	y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])

	rf = _Decision_Tree(max_depth=100)
	rf.fit(X_train, y_train)

	X_test = np.array([[1, 20], [5, 24], [8, 27]])
	preds  = rf.predict(X_test)
	print("Predictions:", preds) 
	print("Expected:   ", [0, 1, 1])


if __name__ == "__main__":
	model = BaseModel(debug=DEBUG)

	# just_majority_class()
	# test_with_int_labels()
	# test_with_string_labels()
	just_test_DecisionTree()



__all__ = ["_Decision_Tree", "RandomForest"]
