from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import RandomForestClassifier  # noqa: E402

DATA_DIRECTORIES = (PROJECT_ROOT / "Data", PROJECT_ROOT / "test")
TARGET_NAMES = ("target", "label", "class", "y", "survived")
MISSING_TOKEN = "__missing__"


@dataclass
class FeatureEncoder:
    column: str
    numeric_fill_value: float | None = None
    category_to_value: dict[str, float] | None = None

    @property
    def is_numeric(self) -> bool:
        return self.category_to_value is None

    def encode(self, raw_value: str | None, source: Path, row_number: int) -> float:
        value = normalize_value(raw_value)

        if self.is_numeric:
            if value == "":
                return (
                    self.numeric_fill_value
                    if self.numeric_fill_value is not None
                    else 0.0
                )
            try:
                return float(value)
            except ValueError as error:
                raise ValueError(
                    f"Feature '{self.column}' in {source} at CSV row {row_number} "
                    f"was numeric in training data but received non-numeric value {value!r}."
                ) from error

        if self.category_to_value is None:
            raise RuntimeError(
                f"Categorical encoder missing for column '{self.column}'."
            )

        key = value or MISSING_TOKEN
        if key not in self.category_to_value:
            self.category_to_value[key] = float(len(self.category_to_value))
        return self.category_to_value[key]


def normalize_value(raw_value: str | None) -> str:
    return "" if raw_value is None else raw_value.strip()


def can_parse_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_data_directory() -> Path:
    for directory in DATA_DIRECTORIES:
        if (directory / "train.csv").exists():
            return directory

    for directory in DATA_DIRECTORIES:
        if directory.exists() and any(directory.glob("*.csv")):
            return directory

    fallback = PROJECT_ROOT / "test"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV file has no header: {path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    return rows, list(fieldnames)


def infer_target_column(columns: list[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for name in TARGET_NAMES:
        if name in lowered:
            return lowered[name]
    return columns[-1]


def split_rows(
    rows: list[dict[str, str]],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not rows:
        raise ValueError("Dataset cannot be empty.")
    if len(rows) == 1:
        return rows, rows

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    test_count = max(1, round(len(rows) * test_size))
    test_count = min(test_count, len(rows) - 1)

    test_indices = set(indices[:test_count])
    train_rows = [row for index, row in enumerate(rows) if index not in test_indices]
    test_rows = [row for index, row in enumerate(rows) if index in test_indices]
    return train_rows, test_rows


def build_feature_encoders(
    rows: list[dict[str, str]],
    feature_columns: list[str],
) -> list[FeatureEncoder]:
    encoders: list[FeatureEncoder] = []

    for column in feature_columns:
        values = [normalize_value(row.get(column)) for row in rows]
        non_empty_values = [value for value in values if value != ""]

        if non_empty_values and all(
            can_parse_float(value) for value in non_empty_values
        ):
            numeric_values = [float(value) for value in non_empty_values]
            encoders.append(
                FeatureEncoder(
                    column=column,
                    numeric_fill_value=median(numeric_values)
                    if numeric_values
                    else 0.0,
                )
            )
            continue

        category_to_value: dict[str, float] = {}
        for value in values:
            key = value or MISSING_TOKEN
            if key not in category_to_value:
                category_to_value[key] = float(len(category_to_value))

        if not category_to_value:
            category_to_value[MISSING_TOKEN] = 0.0

        encoders.append(
            FeatureEncoder(column=column, category_to_value=category_to_value)
        )

    return encoders


def encode_rows(
    rows: list[dict[str, str]],
    encoders: list[FeatureEncoder],
    source: Path,
) -> list[list[float]]:
    X: list[list[float]] = []

    for row_number, row in enumerate(rows, start=2):
        encoded_row = [
            encoder.encode(row.get(encoder.column), source, row_number)
            for encoder in encoders
        ]
        X.append(encoded_row)

    return X


def extract_targets(
    rows: list[dict[str, str]],
    target_column: str,
    source: Path,
) -> list[str]:
    y: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        value = normalize_value(row.get(target_column))
        if value == "":
            raise ValueError(
                f"Missing target value for '{target_column}' in {source} at CSV row {row_number}."
            )
        y.append(value)

    return y


def prepare_training_data(
    rows: list[dict[str, str]],
    columns: list[str],
    target_column: str,
    source: Path,
) -> tuple[list[list[float]], list[str], list[FeatureEncoder]]:
    feature_columns = [column for column in columns if column != target_column]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {source}.")

    encoders = build_feature_encoders(rows, feature_columns)
    X = encode_rows(rows, encoders, source)
    y = extract_targets(rows, target_column, source)
    return X, y, encoders


def select_single_dataset(csv_files: list[Path]) -> Path:
    scored_files: list[tuple[int, Path]] = []
    for path in csv_files:
        _, columns = read_csv(path)
        score = 1 if any(column.lower() in TARGET_NAMES for column in columns) else 0
        scored_files.append((score, path))

    scored_files.sort(key=lambda item: (-item[0], item[1].name))
    return scored_files[0][1]


def load_dataset() -> tuple[list[list[float]], list[str], list[list[float]], str]:
    data_dir = resolve_data_directory()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if train_path.exists():
        train_rows, train_columns = read_csv(train_path)
        target_column = infer_target_column(train_columns)
        X_train_full, y_train_full, encoders = prepare_training_data(
            train_rows,
            train_columns,
            target_column,
            train_path,
        )

        if test_path.exists():
            test_rows, test_columns = read_csv(test_path)
            feature_columns = [
                column for column in train_columns if column != target_column
            ]
            missing_columns = [
                column for column in feature_columns if column not in test_columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing feature columns in {test_path}: {', '.join(missing_columns)}"
                )

            X_test = encode_rows(test_rows, encoders, test_path)
            return (
                X_train_full,
                y_train_full,
                X_test,
                "Loaded "
                f"{relative_to_root(train_path)} + {relative_to_root(test_path)} "
                f"(target column: {target_column})",
            )

        train_rows_split, test_rows_split = split_rows(train_rows)
        X_train, y_train, encoders = prepare_training_data(
            train_rows_split,
            train_columns,
            target_column,
            train_path,
        )
        X_test = encode_rows(test_rows_split, encoders, train_path)
        return (
            X_train,
            y_train,
            X_test,
            "Loaded "
            f"{relative_to_root(train_path)} and performed a simple train/test split "
            f"(target column: {target_column})",
        )

    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        dataset_path = select_single_dataset(csv_files)
        rows, columns = read_csv(dataset_path)
        target_column = infer_target_column(columns)
        train_rows, test_rows = split_rows(rows)
        X_train, y_train, encoders = prepare_training_data(
            train_rows,
            columns,
            target_column,
            dataset_path,
        )
        X_test = encode_rows(test_rows, encoders, dataset_path)
        return (
            X_train,
            y_train,
            X_test,
            "Loaded "
            f"{relative_to_root(dataset_path)} and performed a simple train/test split "
            f"(target column: {target_column})",
        )

    X_train = [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.2],
    ]
    y_train = ["A", "A", "A", "B", "B", "B"]
    X_test = [[0.05, 0.1], [1.05, 1.0]]
    return X_train, y_train, X_test, "Loaded built-in fallback dataset"


def main() -> None:
    X_train, y_train, X_test, source = load_dataset()

    model = RandomForestClassifier(n_trees=50, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(source)
    print(f"Training rows: {len(X_train)}")
    print(f"Prediction rows: {len(X_test)}")
    print("Predictions:")
    for index, prediction in enumerate(predictions, start=1):
        print(f"  {index}: {prediction}")


if __name__ == "__main__":
    main()
