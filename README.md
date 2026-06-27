# Random Forest From Scratch

A clean, reusable Random Forest classifier implemented from scratch in pure Python.

This refactor keeps only the model code needed for training and prediction:
- `RandomForestClassifier`
- `DecisionTreeClassifier`
- split / entropy / information gain helpers

There is no Titanic-specific logic, no preprocessing pipeline, no CSV export pipeline, and no evaluation layer in the core implementation.

## Project Structure

```text
Random_Forest_From_Scratch/
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── test/
│   ├── sample.csv
│   ├── simple_test.py
│   └── test.py
└── README.md
```

## Core Usage

From the project root, import the classifier like this:

```python
from src.model import RandomForestClassifier

model = RandomForestClassifier(n_trees=50)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

You can also use the package export:

```python
from src import RandomForestClassifier
```

## Notes

- The core model in `src/` expects numeric features.
- `test/test.py` includes lightweight CSV preparation for convenience:
  - missing numeric values are filled with the training-column median
  - categorical values are ordinal-encoded
- Labels can be numeric or string values.

## Example

Run the minimal smoke test from the project root:

```bash
python3 test/simple_test.py
```

## Testing With Your Own Data

Drop your dataset into the root `test/` folder.

### Supported patterns

1. **Single CSV file**
   - `test/test.py` loads the first CSV in `test/`
   - it treats `target`, `label`, `class`, or `y` as the target column if present
   - otherwise it uses the last column as the target
   - it performs a simple train/test split automatically

2. **Prepared files**
   - put `train.csv` in `test/` or `Data/` with features + target
   - optionally put `test.csv` in the same folder with matching feature columns
   - if `test.csv` exists, predictions are generated for that file directly

Run:

```bash
python3 test/test.py
```

The script trains the forest and prints predictions.


🔥 Core structure (important)
core/app.py
App class
register routes
core/router.py
route matching logic
core/request.py
request parsing (body, query, headers)
core/response.py
response builder
core/middleware.py
before/after hooks
core/exceptions.py
error handling