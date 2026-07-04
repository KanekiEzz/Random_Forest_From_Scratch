from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import RandomForest


X_train = [
    [0.0, 0.1],
    [0.2, 0.3],
    [0.3, 0.2],
    [1.0, 1.1],
    [1.2, 1.0],
    [1.1, 1.3],
]
y_train = [0, 0, 0, 1, 1, 1]

X_test = [
    [0.1, 0.2],
    [1.1, 1.2],
    [0.9, 1.0],
]


model = RandomForest(n_trees=50, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions)
