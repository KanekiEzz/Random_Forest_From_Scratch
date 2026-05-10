import os
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


class Calculate:
	# def _entropy(self, y):
	# 	if len(y) == 0:
	# 		return 0.0

	# 	counts = np.bincount(y)
	# 	total = len(y)
	# 	entropy = 0.0

	# 	for count in counts.values():
	# 		p = count / total
	# 		if p > 0:
	# 			entropy -= p * math.log2(p)

	# 	return entropy

	def _entropy(self, y):
		if len(y) == 0:
			return 0.0
		
		counts = np.bincount(y)
		
		probs = counts / len(y)
		
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

		if n == 0:
			return 0.0
		
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
		print(f"training model...\nx:{X}, y:{y}")
		self.root = self._grow_tree(X, y, depth=0)

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



# Testing the LabelEncoderSimple
if __name__ == "__main__":
	model = BaseModel(debug=DEBUG)
	# o = ["A", "B", "A", "D", "C"]
	# l = sorted(set(o))
	# model.log(f"llll {l}")


	

	y = ["cat", "dog", "cat", "bird"]
	encoder = LabelEncoderSimple()
	encoder.fit(y)

	y_encoded = encoder.transform(y)
	y_decoder = encoder.inverse_transform(y_encoded)

	print(y_decoder)
	print(y_encoded)


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

	

