![Decision Forests](/public/hi.gif)
# Random Forest From Scratch

A clean, educational implementation of a Random Forest classifier in pure Python with support for both numeric and string labels, and configurable split criteria (entropy/Gini).

### Read the full articles:
- Random Forests from Scratch: https://kanekiezz.vercel.app/posts/random-forests-from-scratch
- Performance Metrics for Regression & Classification: https://kanekiezz.vercel.app/posts/performance-metrics-regression-classification


![Decision Forests](/public/Random_Forest.gif)

## Project Structure

```
Random_Forest_From_Scratch/
├── src/
│   ├── __init__.py                 # Package exports: RandomForest, _Decision_Tree
│   ├── model.py                    # Core implementations
│   ├── test.py                     # Unit tests for components
│   └── test.ipynb                  # Demo: trains RandomForest on the Iris dataset,
│                                    # evaluates with accuracy + confusion matrix heatmap
├── test/
│   ├── simple_test.py              # Minimal working example
│   ├── test.py                     # CSV data loader and full pipeline
│   ├── sample.csv                  # Sample data (feature_1, feature_2, target)
├── Data/
│   ├── train.csv                   # Titanic training set
│   ├── test.csv                    # Titanic test set
│   ├── example_train.csv           # Small example training data
│   └── example_test.csv            # Small example test data
├── Decision_Tree.png               # Decision tree architecture diagram
├── random_forest.png               # Random forest architecture diagram
├── requirements.txt                # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+

### Setup

1. Clone or download the repository
2. Install requirements:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pandas>=2.2.0
matplotlib>=3.9.0
seaborn>=0.13.0
scikit-learn>=1.6.0
```

## Quick Start

### Minimal Example (Hardcoded Data)

```bash
python3 test/simple_test.py
```

This runs the simplest example with in-code training and test data.

### Load and Train on CSV

```bash
python3 test/test.py
```

This automatically loads `Data/train.csv` and `Data/test.csv` (if present), or any CSV in the `test/` folder, and trains the forest.

## Core Usage

### Basic Classification

```python
from src import RandomForest
import numpy as np

# Numeric labels
X_train = np.array([
    [0.0, 0.1],
    [0.2, 0.3],
    [0.3, 0.2],
    [1.0, 1.1],
    [1.2, 1.0],
    [1.1, 1.3],
])
y_train = np.array([0, 0, 0, 1, 1, 1])

model = RandomForest(n_trees=50, random_state=42)
model.fit(X_train, y_train)

X_test = np.array([[0.1, 0.2], [1.1, 1.2]])
predictions = model.predict(X_test)
print(predictions)  # [0 1]
```

### String Labels

```python
from src import RandomForest
import numpy as np

X_train = np.array([
    [1, 10], [2, 11], [3, 12],
    [7, 20], [8, 21], [9, 22],
])
y_train = ["cat", "cat", "cat", "dog", "dog", "dog"]

model = RandomForest(n_trees=100, max_depth=5)
model.fit(X_train, y_train)

X_test = np.array([[2, 11], [8, 21]])
predictions = model.predict(X_test)
print(predictions)  # ['cat' 'dog']
```

## API Reference

### RandomForest

```python
RandomForest(
    n_trees=10,
    max_depth=100,
    min_samples_split=2,
    n_features=None,
    random_state=None,
    criterion="entropy",
    debug=False
)
```

**Parameters:**
- `n_trees` (int, default=10): Number of decision trees in the forest
- `max_depth` (int, default=100): Maximum depth of each tree
- `min_samples_split` (int, default=2): Minimum samples required to split a node
- `n_features` (int or None, default=None): Number of features to consider per split. If None, uses sqrt(total_features)
- `random_state` (int or None, default=None): Seed for reproducibility
- `criterion` (str, default="entropy"): Split criterion, "entropy" or "gini"
- `debug` (bool, default=False): Enable debug logging

**Methods:**

- `fit(X, y)` - Train the forest
  - `X`: 2D array-like of numeric features (n_samples, n_features)
  - `y`: 1D array-like of labels (int or string)

- `predict(X)` - Make predictions
  - `X`: 2D array-like of numeric features (n_samples, n_features)
  - Returns: 1D array of predicted labels (same type as training labels)

---

### _Decision_Tree

Individual decision tree (exported but typically used internally).

```python
_Decision_Tree(
    min_samples_split=2,
    max_depth=100,
    n_features=None,
    rng=None,
    max_thresholds=32,
    criterion="entropy",
    debug=False
)
```

**Parameters:**
- `min_samples_split` (int, default=2): Minimum samples to split
- `max_depth` (int, default=100): Maximum tree depth
- `n_features` (int or None): Number of features to sample per split
- `rng` (np.random.Generator or None): Random number generator
- `max_thresholds` (int, default=32): Maximum number of candidate split thresholds to evaluate
- `criterion` (str, default="entropy"): "entropy" or "gini"
- `debug` (bool, default=False): Enable debug output

**Methods:**

- `fit(X, y)` - Train the tree
- `predict(X)` - Make predictions

---

## Data Format

### Feature Requirements
- **Numeric features only** (int or float)
- Features can be raw numeric values (will be converted to float internally)

### Label Requirements
- **Integer labels**: `[0, 1, 2, ...]` or any int values
- **String labels**: `["cat", "dog", "bird", ...]` or any strings
- Mixed types are automatically detected and encoded internally

### CSV Files
When using `test/test.py` to load CSV data:
- **Target column detection**: Searches for columns named `target`, `label`, `class`, `y`, or `survived` (case-insensitive)
  - If none found, uses the **last column** as target
- **Feature encoding**:
  - Numeric columns: Missing values filled with column median
  - Categorical columns: Automatically ordinal-encoded
- **Train/Test split**:
  - If `train.csv` and `test.csv` both exist: uses both as-is
  - If only `train.csv` exists: performs automatic 80/20 train/test split

## Running Tests

### Unit Tests

```bash
python3 -c "from src.test import Testing; Testing.test_random_forest(); Testing.test_string_labels(); Testing.test_multiclass()"
```

All unit tests in `src/test.py`:
- `test_entropy` - Entropy calculation
- `test_gini` - Gini impurity calculation
- `test_information_gain` - Information gain (entropy-based splitting)
- `test_gini_gain` - Gini gain (Gini-based splitting)
- `test_majority_class` - Majority class voting
- `test_label_encoder` - String label encoding/decoding
- `test_bootstrap` - Bootstrap sampling
- `test_candidate_thresholds` - Threshold generation
- `test_decision_tree` - Single decision tree
- `test_random_forest` - Forest with numeric labels
- `test_string_labels` - Forest with string labels
- `test_multiclass` - Multi-class classification
- `test_xor` - XOR problem (non-linearly separable)

### Minimal Example

```bash
python3 test/simple_test.py
```

Trains a forest on 6 hardcoded samples and predicts on 3 test samples.

### CSV Pipeline

```bash
python3 test/test.py
```

Loads CSV data from `Data/train.csv` (and `Data/test.csv` if present), or any CSV in `test/`, trains the forest, and prints predictions.

### Iris Dataset Demo (Jupyter Notebook)

`src/test.ipynb` demonstrates the classifier on the classic Iris dataset:
- Loads Iris via `sklearn.datasets.load_iris()`
- Splits data (80/20) and standardizes features
- Trains `RandomForest(n_trees=100, random_state=42)`
- Evaluates with accuracy score and a confusion matrix heatmap (seaborn)

Open with:
```bash
jupyter notebook src/test.ipynb
```

## Architecture Diagrams

### Decision Tree

![Decision Tree Architecture](/public/Decision_Tree.png)

Shows the structure of a single decision tree with nodes and splits — each internal node represents a feature/threshold test, leaves represent predicted classes.

### Random Forest

![Random Forest Architecture](/public/random_forest.png)

Illustrates the ensemble approach: multiple decision trees trained on bootstrap samples, combined through majority voting.

## Supported Criteria

The `criterion` parameter controls the split criterion:

- **"entropy"** (default) - Uses information gain to select splits that maximize information gain
  - Formula: IG = H(parent) - weighted H(children)
  - Where H(Y) = -Σ(p_i · log₂(p_i))

- **"gini"** - Uses Gini impurity to select splits
  - Formula: Gini Gain = Gini(parent) - weighted Gini(children)
  - Where Gini(Y) = 1 - Σ(p_i²)

Both criteria perform similarly in practice; Gini is slightly faster computationally.

## Implementation Notes

### Bootstrap Sampling
Each tree in the forest is trained on a bootstrap sample (random sampling with replacement) of the full training set. This introduces diversity among trees and enables out-of-bag error estimation.

### Feature Sampling
At each split, only a random subset of features is considered. By default, this subset size is √(n_features), which is a recommended heuristic to reduce correlation between trees.

### Label Encoding
String labels are automatically encoded to integers during training and decoded back to strings during prediction, making the API transparent to the user.

## Files & Folders

| File/Folder | Purpose |
|---|---|
| `src/model.py` | Core RandomForest and _Decision_Tree implementations |
| `src/test.py` | Unit test suite for all components |
| `src/__init__.py` | Package exports |
| `test/simple_test.py` | Minimal hardcoded example |
| `test/test.py` | CSV loader and full training pipeline |
| `Data/train.csv`, `Data/test.csv` | Large Titanic dataset |
| `Data/example_*.csv` | Small example datasets |
| `test/sample.csv` | Sample features for testing |
| `Decision_Tree.png` | Single tree architecture diagram |
| `random_forest.png` | Forest ensemble architecture diagram |

## Troubleshooting

### Import Errors

If `from src import RandomForest` fails:

```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src import RandomForest
```

Both approaches are supported:
```python
from src import RandomForest           # Package import
from src.model import RandomForest      # Direct import
```

### CSV Loading Issues

The `test/test.py` loader is flexible:
- If it can't find a target column, it uses the last column
- String values in numeric columns are encoded on-the-fly
- Missing numeric values are filled with the column median

### Empty Splits

The algorithm skips splits that would separate zero samples to one side, preventing invalid trees.


## References & Credits

This is an educational implementation demonstrating core Random Forest concepts:
- Bootstrap aggregating for variance reduction
- Random feature subsets for tree diversity
- Majority voting for ensemble predictions
- Support for both entropy and Gini criteria

**Inspiration:** This project was inspired by [Random Forests From Scratch](https://carbonati.github.io/posts/random-forests-from-scratch/) by Carbonati, which walks through the core concepts of building a Random Forest without relying on scikit-learn.

## Notes

- Requirements list includes `scikit-learn` for compatibility; the core implementation uses only NumPy
- The implementation is designed for clarity and understanding, not production performance
- For the Titanic dataset in `Data/`, the preprocessing is automatic in `test/test.py`

