"""
attention_layer.py
──────────────────
Custom Keras attention weight layer used in BiLSTM variants.
Also provides a NumPy-based soft-attention for score fusion.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    """
    Bahdanau-style self-attention layer.
    Input shape : (batch, timesteps, features)
    Output shape: (batch, features)  — context vector
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # Score each timestep
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, T, 1)
        a = tf.nn.softmax(e, axis=1)                    # attention weights
        context = tf.reduce_sum(x * a, axis=1)          # (batch, features)
        return context

    def get_config(self):
        return super().get_config()


# ── NumPy soft-attention for ensemble fusion ───────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def attention_fusion(scores: np.ndarray, temperatures: np.ndarray = None) -> float:
    """
    Soft-attention weighted fusion of multiple model probability scores.

    Parameters
    ----------
    scores       : 1-D array of probabilities from each model
    temperatures : optional weighting temperatures (higher = more weight)

    Returns
    -------
    float — fused probability
    """
    if temperatures is None:
        temperatures = np.ones(len(scores))
    weights = softmax(temperatures)
    return float(np.dot(weights, scores))
