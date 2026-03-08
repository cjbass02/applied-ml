from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28)
INPUT_SHAPE = (28, 28, 1)
CLASS_NAMES = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"


FULL_MLP_CONFIG = {
    "hidden_layers": [0, 1, 2, 3],
    "unit_pairs": [(8, 8), (32, 32), (64, 32)],
    "optimizers": ["adam", "rmsprop"],
    "epochs": [10, 15],
}

FULL_CNN_CONFIG = {
    "conv_options": [
        (16, 32, 3),
        (32, 64, 3),
        (32, 64, 5),
    ],
    "use_pooling": [False, True],
    "optimizers": ["adam", "rmsprop"],
    "epochs": [10, 15],
}


@dataclass
class ExperimentResult:
    model_family: str
    config_name: str
    optimizer: str
    epochs: int
    best_epoch: int
    train_accuracy: float
    val_accuracy: float
    best_val_accuracy: float
    test_accuracy: float
    loss: float
    params: int
    model_path: str
    summary_path: str
    history_path: str


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_output_dirs() -> None:
    for directory in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, SUMMARIES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_array(path: Path) -> np.ndarray:
    return np.load(path)["arr_0"]


def load_kmnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = load_array(DATA_DIR / "kmnist-train-imgs.npz")
    train_labels = load_array(DATA_DIR / "kmnist-train-labels.npz")
    test_images = load_array(DATA_DIR / "kmnist-test-imgs.npz")
    test_labels = load_array(DATA_DIR / "kmnist-test-labels.npz")
    return train_images, train_labels, test_images, test_labels


def save_dataset_summary(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    summary = {
        "train_images_shape": list(train_images.shape),
        "train_labels_shape": list(train_labels.shape),
        "test_images_shape": list(test_images.shape),
        "test_labels_shape": list(test_labels.shape),
        "num_classes": NUM_CLASSES,
        "image_shape": list(IMAGE_SHAPE),
        "class_names": CLASS_NAMES,
        "classification_type": "multiclass",
        "recommended_loss": "sparse_categorical_crossentropy",
        "default_metric": "accuracy",
    }
    with open(METRICS_DIR / "dataset_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def plot_sample_grid(images: np.ndarray, labels: np.ndarray, class_ids: Iterable[int] = (0, 1, 2, 3)) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(7.5, 9))
    fig.suptitle("KMNIST Sample Images: 3 Examples from 4 Classes", fontsize=14)

    for row, class_id in enumerate(class_ids):
        class_indices = np.where(labels == class_id)[0][:3]
        for col, sample_index in enumerate(class_indices):
            axes[row, col].imshow(images[sample_index], cmap="gray")
            axes[row, col].axis("off")
            axes[row, col].set_title(f"class {class_id}: {CLASS_NAMES[class_id]}")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sample_grid_4_classes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    return images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0


def make_train_validation_split(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float = 0.1,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0].copy()
        rng.shuffle(class_indices)
        val_count = int(len(class_indices) * validation_fraction)
        val_indices.append(class_indices[:val_count])
        train_indices.append(class_indices[val_count:])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return images[train_idx], images[val_idx], labels[train_idx], labels[val_idx]


def get_model_summary_text(model: tf.keras.Model) -> str:
    buffer = StringIO()
    model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
    return buffer.getvalue()


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_history_csv(path: Path, history: tf.keras.callbacks.History) -> None:
    keys = list(history.history.keys())
    epochs = len(history.history[keys[0]])
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for epoch_index in range(epochs):
            writer.writerow([epoch_index + 1, *[history.history[key][epoch_index] for key in keys]])


def plot_training_history(history: tf.keras.callbacks.History, title: str, path: Path) -> None:
    epochs = np.arange(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history.history["accuracy"], label="train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="validation")
    axes[0].set_title(f"{title} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="train")
    axes[1].plot(epochs, history.history["val_loss"], label="validation")
    axes[1].set_title(f"{title} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(confusion: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    threshold = confusion.max() / 2 if confusion.size else 0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            color = "white" if confusion[row, col] > threshold else "black"
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_mlp(hidden_layers: int, unit_pair: tuple[int, int], optimizer: str) -> tf.keras.Sequential:
    model = tf.keras.Sequential(name=f"mlp_h{hidden_layers}_{unit_pair[0]}_{unit_pair[1]}_{optimizer}")
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Flatten())

    if hidden_layers >= 1:
        model.add(tf.keras.layers.Dense(unit_pair[0], activation="relu"))
    if hidden_layers >= 2:
        model.add(tf.keras.layers.Dense(unit_pair[1], activation="relu"))
    if hidden_layers >= 3:
        model.add(tf.keras.layers.Dense(unit_pair[1], activation="relu"))

    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(
    conv1_filters: int,
    conv2_filters: int,
    kernel_size: int,
    use_pooling: bool,
    unit_pair: tuple[int, int],
    optimizer: str,
) -> tf.keras.Sequential:
    model = tf.keras.Sequential(
        name=f"cnn_{conv1_filters}_{conv2_filters}_k{kernel_size}_{'pool' if use_pooling else 'nopool'}_{optimizer}"
    )
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Conv2D(conv1_filters, kernel_size=(kernel_size, kernel_size), activation="relu", padding="same"))
    model.add(tf.keras.layers.Conv2D(conv2_filters, kernel_size=(kernel_size, kernel_size), activation="relu", padding="same"))
    if use_pooling:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(unit_pair[0], activation="relu"))
    model.add(tf.keras.layers.Dense(unit_pair[1], activation="relu"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    per_class_accuracy = []

    for class_id in range(NUM_CLASSES):
        class_total = confusion[class_id].sum()
        accuracy = float(confusion[class_id, class_id] / class_total) if class_total else 0.0
        per_class_accuracy.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "accuracy": accuracy,
                "correct": int(confusion[class_id, class_id]),
                "total": int(class_total),
            }
        )

    return {
        "overall_accuracy": float(np.mean(y_true == y_pred)),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    model: tf.keras.Model,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    epochs: int,
    batch_size: int = 128,
) -> tf.keras.callbacks.History:
    return model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )


def run_single_experiment(
    model_family: str,
    config_name: str,
    model: tf.keras.Model,
    optimizer: str,
    epochs: int,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> ExperimentResult:
    history = train_model(model, train_images, train_labels, val_images, val_labels, epochs=epochs)
    loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    val_acc_history = history.history["val_accuracy"]
    best_epoch = int(np.argmax(val_acc_history) + 1)
    best_val_accuracy = float(np.max(val_acc_history))
    final_train_accuracy = float(history.history["accuracy"][-1])
    final_val_accuracy = float(val_acc_history[-1])

    model_path = MODELS_DIR / f"{config_name}.keras"
    summary_path = SUMMARIES_DIR / f"{config_name}_summary.txt"
    history_path = METRICS_DIR / f"{config_name}_history.csv"

    model.save(model_path)
    save_text(summary_path, get_model_summary_text(model))
    save_history_csv(history_path, history)

    return ExperimentResult(
        model_family=model_family,
        config_name=config_name,
        optimizer=optimizer,
        epochs=epochs,
        best_epoch=best_epoch,
        train_accuracy=final_train_accuracy,
        val_accuracy=final_val_accuracy,
        best_val_accuracy=best_val_accuracy,
        test_accuracy=float(test_accuracy),
        loss=float(loss),
        params=model.count_params(),
        model_path=str(model_path.relative_to(PROJECT_DIR)),
        summary_path=str(summary_path.relative_to(PROJECT_DIR)),
        history_path=str(history_path.relative_to(PROJECT_DIR)),
    )


def write_experiment_table(path: Path, rows: list[ExperimentResult]) -> None:
    fieldnames = list(ExperimentResult.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run_mlp_search(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config: dict[str, list],
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for hidden_layers in config["hidden_layers"]:
        for unit_pair in config["unit_pairs"]:
            for optimizer in config["optimizers"]:
                for epochs in config["epochs"]:
                    config_name = f"mlp_h{hidden_layers}_u{unit_pair[0]}_{unit_pair[1]}_{optimizer}_e{epochs}"
                    model = build_mlp(hidden_layers=hidden_layers, unit_pair=unit_pair, optimizer=optimizer)
                    result = run_single_experiment(
                        model_family="mlp",
                        config_name=config_name,
                        model=model,
                        optimizer=optimizer,
                        epochs=epochs,
                        train_images=train_images,
                        train_labels=train_labels,
                        val_images=val_images,
                        val_labels=val_labels,
                        test_images=test_images,
                        test_labels=test_labels,
                    )
                    results.append(result)

    return results


def run_cnn_search(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    dense_units: tuple[int, int],
    config: dict[str, list],
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for conv1_filters, conv2_filters, kernel_size in config["conv_options"]:
        for use_pooling in config["use_pooling"]:
            for optimizer in config["optimizers"]:
                for epochs in config["epochs"]:
                    config_name = (
                        f"cnn_c{conv1_filters}_{conv2_filters}_k{kernel_size}_"
                        f"{'pool' if use_pooling else 'nopool'}_{optimizer}_e{epochs}"
                    )
                    model = build_cnn(
                        conv1_filters=conv1_filters,
                        conv2_filters=conv2_filters,
                        kernel_size=kernel_size,
                        use_pooling=use_pooling,
                        unit_pair=dense_units,
                        optimizer=optimizer,
                    )
                    result = run_single_experiment(
                        model_family="cnn",
                        config_name=config_name,
                        model=model,
                        optimizer=optimizer,
                        epochs=epochs,
                        train_images=train_images,
                        train_labels=train_labels,
                        val_images=val_images,
                        val_labels=val_labels,
                        test_images=test_images,
                        test_labels=test_labels,
                    )
                    results.append(result)

    return results


def select_best_result(results: list[ExperimentResult]) -> ExperimentResult:
    return max(results, key=lambda item: (item.best_val_accuracy, item.val_accuracy, -item.loss))


def parse_mlp_units(config_name: str) -> tuple[int, int]:
    token = config_name.split("_")
    first_units = int(token[2].removeprefix("u"))
    second_units = int(token[3])
    return first_units, second_units


def rebuild_best_model(best_result: ExperimentResult) -> tf.keras.Model:
    model_path = PROJECT_DIR / best_result.model_path
    return tf.keras.models.load_model(model_path)


def save_best_model_evaluation(
    best_result: ExperimentResult,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    model = rebuild_best_model(best_result)
    predictions = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    evaluation = evaluate_predictions(test_labels, predicted_labels)

    metrics_path = METRICS_DIR / f"{best_result.config_name}_evaluation.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)

    confusion = np.array(evaluation["confusion_matrix"])
    plot_confusion_matrix(
        confusion,
        FIGURES_DIR / f"{best_result.config_name}_confusion_matrix.png",
        title=f"Confusion Matrix: {best_result.config_name}",
    )


def save_best_history_plot(best_result: ExperimentResult) -> None:
    history_path = PROJECT_DIR / best_result.history_path
    data = np.genfromtxt(history_path, delimiter=",", names=True)
    history = {
        "accuracy": data["accuracy"].tolist(),
        "val_accuracy": data["val_accuracy"].tolist(),
        "loss": data["loss"].tolist(),
        "val_loss": data["val_loss"].tolist(),
    }

    history_wrapper = type("HistoryWrapper", (), {"history": history})()
    plot_training_history(
        history_wrapper,
        title=best_result.config_name,
        path=FIGURES_DIR / f"{best_result.config_name}_history.png",
    )


def build_run_summary(
    mlp_results: list[ExperimentResult],
    cnn_results: list[ExperimentResult],
) -> str:
    best_mlp = select_best_result(mlp_results)
    best_cnn = select_best_result(cnn_results)

    lines = [
        "Project 3 KMNIST experiment run",
        "",
        "Classification type: multiclass",
        "Recommended loss: sparse_categorical_crossentropy",
        "Reason for 10 output nodes: one softmax probability per KMNIST class",
        "",
        f"MLP experiments run: {len(mlp_results)}",
        f"Best MLP config: {best_mlp.config_name}",
        f"Best MLP validation accuracy: {best_mlp.best_val_accuracy:.4f}",
        f"Best MLP test accuracy: {best_mlp.test_accuracy:.4f}",
        "",
        f"CNN experiments run: {len(cnn_results)}",
        f"Best CNN config: {best_cnn.config_name}",
        f"Best CNN validation accuracy: {best_cnn.best_val_accuracy:.4f}",
        f"Best CNN test accuracy: {best_cnn.test_accuracy:.4f}",
        "",
        "Saved outputs:",
        "- figures/: sample grid, confusion matrices, training curves",
        "- metrics/: dataset summary, experiment tables, evaluation JSON, training histories",
        "- models/: saved Keras models",
        "- summaries/: model architecture summaries",
    ]
    return "\n".join(lines)


def main() -> None:
    set_global_seed()
    ensure_output_dirs()

    train_images_raw, train_labels, test_images_raw, test_labels = load_kmnist_data()
    save_dataset_summary(train_images_raw, train_labels, test_images_raw, test_labels)
    plot_sample_grid(train_images_raw, train_labels)

    train_images = preprocess_images(train_images_raw)
    test_images = preprocess_images(test_images_raw)
    train_images, val_images, train_labels, val_labels = make_train_validation_split(train_images, train_labels)

    mlp_results = run_mlp_search(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
        config=FULL_MLP_CONFIG,
    )
    write_experiment_table(METRICS_DIR / "mlp_experiment_results.csv", mlp_results)

    best_mlp = select_best_result(mlp_results)
    best_dense_units = parse_mlp_units(best_mlp.config_name)
    save_best_model_evaluation(best_mlp, test_images, test_labels)
    save_best_history_plot(best_mlp)

    cnn_results = run_cnn_search(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
        dense_units=best_dense_units,
        config=FULL_CNN_CONFIG,
    )
    write_experiment_table(METRICS_DIR / "cnn_experiment_results.csv", cnn_results)

    best_cnn = select_best_result(cnn_results)
    save_best_model_evaluation(best_cnn, test_images, test_labels)
    save_best_history_plot(best_cnn)

    summary_text = build_run_summary(mlp_results, cnn_results)
    save_text(OUTPUT_DIR / "run_summary.txt", summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
