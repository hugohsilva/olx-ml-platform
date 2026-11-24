from __future__ import annotations

import csv
import json
import os
from pathlib import Path

FEATURE_COLUMNS = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)


def load_dataset() -> tuple[list[list[float]], list[int]]:
    data_path = Path(__file__).parent / "data" / "iris.csv"
    features: list[list[float]] = []
    labels: list[int] = []
    species_to_label = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }

    with data_path.open(newline="", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file)
        if reader.fieldnames != [*FEATURE_COLUMNS, "species"]:
            raise ValueError("The Iris CSV does not match the expected five-column schema.")
        for row_number, row in enumerate(reader, start=2):
            try:
                feature_row = [float(row[column]) for column in FEATURE_COLUMNS]
                label = species_to_label[row["species"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid Iris record on line {row_number}.") from exc
            features.append(feature_row)
            labels.append(label)

    if not features:
        raise ValueError("The Iris dataset is empty.")
    return features, labels


def main() -> None:
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    features, labels = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=7,
        stratify=labels,
    )
    model = LogisticRegression(max_iter=200, random_state=7)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    output_directory = Path(os.environ["MLPLATFORM_OUTPUT_DIR"])
    output_directory.mkdir(parents=True, exist_ok=True)
    dump(model, output_directory / "model.joblib")
    (output_directory / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                "test_samples": len(y_test),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output_directory / "parameters.json").write_text(
        json.dumps(
            {
                "max_iter": model.max_iter,
                "random_state": 7,
                "test_size": 0.2,
                "training_samples": len(y_train),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
