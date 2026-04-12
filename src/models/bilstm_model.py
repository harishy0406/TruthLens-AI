"""
bilstm_model.py
───────────────
Bidirectional LSTM for text classification.
Architecture: Embedding → BiLSTM → Dense → Output
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout, SpatialDropout1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report

VOCAB_SIZE = 4000
MAX_LEN    = 120
EMBED_DIM  = 64


class BiLSTMModel:
    """Bidirectional LSTM text classifier."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = MAX_LEN, embed_dim: int = EMBED_DIM):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.embed_dim  = embed_dim
        self.model      = self._build()
        self.history    = None

    def _build(self) -> tf.keras.Model:
        model = Sequential([
            Embedding(self.vocab_size, self.embed_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
            Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.1)),
            Dense(64, activation="relu"),
            Dropout(0.4),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val,
            epochs: int = 15, batch_size: int = 128):
        print("[BiLSTM] Training …")
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
        print(f"[BiLSTM] Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        return {"accuracy": acc, "report": report, "y_pred": y_pred,
                "history": self.history.history if self.history else {}}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[BiLSTM] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BiLSTMModel":
        obj = cls()
        obj.model = load_model(path)
        return obj
