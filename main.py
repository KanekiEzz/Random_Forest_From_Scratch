from ensurepip import bootstrap
from itertools import count
import os
from statistics import mode
from sys import modules
import numpy as np
import math
from collections import Counter


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


class Calculate(BaseModel):
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
		
		return (
					(len(y_left) / n) * self._entropy(y_left) +
					(len(y_right) / n) * self._entropy(y_right)
				)

	# IG = H_parent = H_after
	def information_gain(self, y_parent: np.array, y_left: np.array, y_right: np.array) -> float:
		print("how much entropy after use split")
		return self._entropy(y_parent) - self.H_after(y_left, y_right)

	"""
		Creates a bootstrap sample from X and y using random sampling with replacement,
		and returns the sampled data along with out-of-bag samples.
	"""
	def bootstrap_sample(self, X: np.array, y: np.array) ->  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		n_samples = X.shape[0]
		self.log(f"n_samples: {n_samples}")
		
		
		idxs = np.random.choice(n_samples, n_samples, replace=True)

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
		debug: bool = DEBUG
	) -> None:
		super().__init__(debug)
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_features = n_features
		self.root = None
		self.calc = Calculate(debug=debug)

	def fit(self, X, y) -> None:
		print(f"training model...\nx:{X}, y:{y}")
		self.root = self._grow_tree(X, y, depth=0)

	def predict(self, X) -> None:
		print("predict")

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
		# best_feature, best_threshold = self._Find_Best_Split(X, y)
	
		# if no split found
		# if best_feature is None:
		# 	leaf_value = Calculate()._Majority_class(y)
		# 	return _Tree_Node(value=leaf_value)
	
		# # split using numpy masks
		# left_idxs = X[:, best_feature] <= best_threshold
		# right_idxs = X[:, best_feature] > best_threshold
	
		# X_left = X[left_idxs]
		# y_left = y[left_idxs]
	
		# X_right = X[right_idxs]
		# y_right = y[right_idxs]
	
		# recursive grow
		# left = self._grow_tree(X_left, y_left, depth + 1)
		# right = self._grow_tree(X_right, y_right, depth + 1)
	
		# return _Tree_Node(
	 #        feature=best_feature,
	 #        threshold=best_threshold,
	 #        left=left,
	 #        right=right
	 #    )

	def _Calculate_Entropy(self) -> None:
		print("calculate grow tree")

	def _traverse(self, x, node) -> None:
		return None

	def _Find_Best_Split(self, X, y) -> None:
		print("find best split")




class RandomForest(BaseModel):
	def __init__(self, n_trees=10, debug: bool = DEBUG):
		super().__init__(debug)
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
			self.log("training model...", "INFO")
		pass

	def _bootstrap_sample(self, X: np.array, y: np.array):
		return self.call.bootstrap_sample(np.array(X), np.array(y))

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



# Testing the LabelEncoderSimple
if __name__ == "__main__":
	model = BaseModel(debug=DEBUG)
	# o = ["A", "B", "A", "D", "C"]
	# l = sorted(set(o))
	# model.log(f"llll {l}")


	

	# test Label encoder simple
	# y = ["cat", "dog", "cat", "bird"]
	# encoder = LabelEncoderSimple()
	# encoder.fit(y)
	# y_encoded = encoder.transform(y)
	# y_decoder = encoder.inverse_transform(y_encoded)
	# print(y_decoder)
	# print(y_encoded)


	#test bincount
	# y = [1, 1, 2 ,2]
	# y = [0,0,1,1]
	# y = ["cat", "dog", "cat", "bird"]
	# encoder = LabelEncoderSimple()
	# encoder.fit(y)
	# y_encoder = encoder.transform(y)
	# count = np.bincount(y_encoder)
	# model.log(f"{count}")
	# len = len(y)
	# model.log(f"len: {len}")


	# test Calculate entropy
	# y = [1, 1, 2, 2]
	# calc = Calculate(y)
	# calc._entropy(y)

	#test bootstrap_sample
	# x = [1, 25, 3, 40 ,10]
	# y = [2, 22, 33, 6, 9]

	# bootstrap = RandomForest()

	# X_sample, y_sample, X_oob, y_oob = bootstrap._bootstrap_sample(x, y)
	# model.log(f"x: {x}")
	# model.log(f"y: {y}")
	# model.log(f"X_sample: {X_sample}", "info")
	# model.log(f"y_sample: {y_sample}", "info")
	# model.log(f"X_oob: { X_oob,}", "info")
	# model.log(f"y_oob: { y_oob,}", "info")

	# test majority Class
	# y = [0, 1, 1, 1, 0]
	# majority = Calculate()
	# p = majority._Majority_class(y)
	# model.log(f"P: {p}")

	# model.log(f"this a bage value: {y[p]}")


	# test grow tree
	X = np.array([
        [1, 20],
        [2, 21],
        [3, 22],
        [4, 23],
        [5, 24],
        [6, 25]
    ])

	y = np.array([0, 0, 0, 1, 1, 1])

	tree = _Decision_Tree()

	tree.fit(X, y)

	print("\nROOT NODE")
	print("feature:", tree.root.feature)
	print("threshold:", tree.root.threshold)

	print("\nLEFT NODE")
	print(tree.root.left.value)

	print("\nRIGHT NODE")
	print(tree.root.right.value)



# testing entopy calculation
# if __name__ == "__main__":
# 	print("Testing entropy calculation...")
# 	p = Calculate()
# 	y = np.array([0, 0, 1, 1, 1])
# 	print(p._entropy(y))
 
# 	l = Counter(['A', 'B', 'A', 'C', 'B', 'B'])
# 	print(l)
	


# if __name__ == "__main__":

#     np.random.seed(42)

#     # -------------------------
#     # SIMPLE DATASET (toy data)
#     # -------------------------
#     X = np.array([
#         [1, 20],
#         [2, 21],
#         [3, 22],
#         [4, 23],
#         [5, 24],
#         [6, 25],
#         [7, 26],
#         [8, 27]
#     ])

#     y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

#     # -------------------------
#     # CREATE MODEL
#     # -------------------------
#     model = RandomForest(n_trees=5)

#     # -------------------------
#     # TRAIN
#     # -------------------------
#     print("\n🔥 Training Random Forest...")
#     model.fit(X, y)

#     # -------------------------
#     # TEST DATA
#     # -------------------------
#     X_test = np.array([
#         [1, 20],
#         [5, 24],
#         [8, 27]
#     ])

#     # -------------------------
#     # PREDICT
#     # -------------------------
#     print("\n🔮 Predictions...")
#     preds = model.predict(X_test)

#     print("Final Predictions:", preds)

	

