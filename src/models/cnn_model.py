"""
cnn_model.py
────────────
1-D Convolutional Neural Network for text classification.
Architecture: Embedding → Conv1D × 3 → GlobalMaxPool → Dense → Output
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report

VOCAB_SIZE = 4000
MAX_LEN    = 120
EMBED_DIM  = 64


class CNNModel:
    """Text CNN with three parallel convolutional filter sizes."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = MAX_LEN, embed_dim: int = EMBED_DIM):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.embed_dim  = embed_dim
        self.model      = self._build()
        self.history    = None

    def _build(self) -> tf.keras.Model:
        model = Sequential([
            Embedding(self.vocab_size, self.embed_dim, input_length=self.max_len),
            Conv1D(128, 3, activation="relu", padding="same"),
            BatchNormalization(),
            Conv1D(128, 4, activation="relu", padding="same"),
            BatchNormalization(),
            Conv1D(64, 5, activation="relu", padding="same"),
            GlobalMaxPooling1D(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val,
            epochs: int = 15, batch_size: int = 256):
        print("[CNN] Training …")
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    def predict(self, X) -> np.ndarray:
        probs = self.model.predict(X, verbose=0).flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"[CNN] Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        return {"accuracy": acc, "report": report, "y_pred": y_pred,
                "history": self.history.history if self.history else {}}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[CNN] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CNNModel":
        obj = cls()
        obj.model = load_model(path)
        return obj
