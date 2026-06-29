import numpy as np

try:
	from .model import (
		Calculate,
		Utils,
		LabelEncoderSimple,
		_Decision_Tree,
		RandomForest,
		BaseModel,
		DEBUG
	)
except ImportError:
	from model import (
		Calculate,
		Utils,
		LabelEncoderSimple,
		_Decision_Tree,
		RandomForest,
		BaseModel,
		DEBUG
	)

class Testing:
	# 1. Entropy
	@staticmethod
	def test_entropy():
		calc = Calculate()

		y = np.array([0, 0, 0, 0])
		assert calc._entropy(y) == 0.0

		y = np.array([0, 1])
		assert np.isclose(calc._entropy(y), 1.0)

		y = np.array([0, 0, 1, 1])
		assert np.isclose(calc._entropy(y), 1.0)

		print("Entropy tests passed")

	# 2. Gini
	@staticmethod
	def test_gini():
		calc = Calculate()

		y = np.array([0, 0, 0])
		assert calc.gini(y) == 0

		y = np.array([0, 1])
		assert np.isclose(calc.gini(y), 0.5)

		y = np.array([0, 0, 1, 1])
		assert np.isclose(calc.gini(y), 0.5)

		print("Gini tests passed")

	# 3. Information Gain
	@staticmethod
	def test_information_gain():
		calc = Calculate()

		y_parent = np.array([0,0,1,1])
		y_left = np.array([0,0])
		y_right = np.array([1,1])

		gain = calc.information_gain(
			y_parent,
			y_left,
			y_right
		)

		assert np.isclose(gain,1.0)

		print("Information Gain passed")

	# 4. Gini Gain
	@staticmethod
	def test_gini_gain():
		calc = Calculate()

		y_parent = np.array([0,0,1,1])
		y_left = np.array([0,0])
		y_right = np.array([1,1])

		gain = calc.gini_gain(
			y_parent,
			y_left,
			y_right
		)

		assert np.isclose(gain,0.5)

		print("Gini Gain passed")

	# 5. Majority Class
	@staticmethod
	def test_majority_class():
		calc = Calculate()

		y = np.array([0,1,1,1])

		assert calc._Majority_class(y) == 1

		print("Majority class passed")

	# 6. Label Encoder
	@staticmethod
	def test_label_encoder():
		encoder = LabelEncoderSimple()

		y = ["cat","dog","bird","cat"]

		encoder.fit(y)

		encoded = encoder.transform(y)

		decoded = encoder.inverse_transform(encoded)

		assert list(decoded) == y

		print("Label encoder passed")

	# 7. Bootstrap
	@staticmethod
	def test_bootstrap():
		calc = Calculate()

		rng = np.random.default_rng(42)

		X = np.arange(20).reshape(10,2)
		y = np.arange(10)

		X_sample,y_sample,X_oob,y_oob = calc.bootstrap_sample(X,y,rng)

		assert len(X_sample)==10
		assert len(y_sample)==10

		print("Bootstrap passed")

	# 8. Candidate Thresholds
	@staticmethod
	def test_candidate_thresholds():
		values = np.array([1,2,3,4])

		thresholds = Utils.candidate_thresholds(values)

		expected = np.array([1.5,2.5,3.5])

		assert np.allclose(thresholds,expected)

		print("Threshold test passed")

	# 9. Decision Tree
	@staticmethod
	def test_decision_tree():
		X = np.array([
			[1],
			[2],
			[3],
			[10],
			[11],
			[12]
		])

		y = np.array([
			0,
			0,
			0,
			1,
			1,
			1
		])

		tree = _Decision_Tree(
			max_depth=5
		)

		tree.fit(X,y)

		preds = tree.predict(X)

		assert np.array_equal(preds,y)

		print("Decision Tree passed")

	# 10. Random Forest
	@staticmethod
	def test_random_forest():
		X = np.array([
			[1],
			[2],
			[3],
			[10],
			[11],
			[12]
		])

		y = np.array([
			0,
			0,
			0,
			1,
			1,
			1
		])

		rf = RandomForest(
			n_trees=20,
			random_state=42
		)

		rf.fit(X,y)

		preds = rf.predict(X)

		accuracy = np.mean(preds==y)

		assert accuracy==1.0

		print("Random Forest passed")

	# 11. String Labels
	@staticmethod
	def test_string_labels():
		X = np.array([
			[1],
			[2],
			[10],
			[11]
		])

		y = np.array([
			"cat",
			"cat",
			"dog",
			"dog"
		])

		rf = RandomForest(
			n_trees=20,
			random_state=42
		)

		rf.fit(X,y)

		preds = rf.predict(X)

		assert list(preds)==list(y)

		print("String labels passed")

	# 12. Multi-class
	@staticmethod
	def test_multiclass():
		X = np.array([
			[1],
			[2],
			[10],
			[11],
			[20],
			[21]
		])

		y = np.array([
			0,
			0,
			1,
			1,
			2,
			2
		])

		rf = RandomForest(
			n_trees=30,
			random_state=42
		)

		rf.fit(X,y)

		preds = rf.predict(X)

		assert np.array_equal(preds,y)

		print("Multiclass passed")

	# 13. XOR Test (hard problem)
	@staticmethod
	def test_xor():
		X = np.array([
			[0,0],
			[0,1],
			[1,0],
			[1,1]
		])

		y = np.array([
			0,
			1,
			1,
			0
		])

		rf = RandomForest(
			n_trees=100,
			max_depth=5,
			random_state=42
		)

		rf.fit(X,y)

		preds = rf.predict(X)

		assert np.array_equal(preds,y)

		print("XOR passed")
		
		
# Testing	
if __name__ == "__main__":
	model = BaseModel(debug=DEBUG)
    
	test = Testing()
	test.test_entropy()
	test.test_gini()
	test.test_information_gain()
	test.test_gini_gain()
	test.test_majority_class()
	test.test_label_encoder()
	test.test_bootstrap()
	test.test_candidate_thresholds()
	test.test_decision_tree()
	test.test_random_forest()
	test.test_string_labels()
	test.test_multiclass()
	test.test_xor()
