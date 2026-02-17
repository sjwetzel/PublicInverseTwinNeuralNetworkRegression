# -*- coding: utf-8 -*-
"""
MDN with generator-based training (no noise), stable validation evaluation
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from data_utils import (
    num_data_points,
    ground_truth,
    sample_data,
    enforce_boundaries
)

# =========================
# Split Data
# =========================
def split(x, y, val_pct=0.2, test_pct=0.2, seed=None):
    n = len(x)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_pct)
    n_val = int(n * val_pct)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (x[train_idx], y[train_idx]), \
           (x[val_idx], y[val_idx]), \
           (x[test_idx], y[test_idx])


# =========================
# MDN Loss
# =========================
def mdn_loss(n_components, output_dim):
    def loss_fn(y_true, y_pred):
        logits = y_pred[:, :n_components]
        means = y_pred[:, n_components:n_components + n_components * output_dim]
        log_sigmas = y_pred[:, n_components + n_components * output_dim:]

        means = tf.reshape(means, [-1, n_components, output_dim])
        log_sigmas = tf.reshape(log_sigmas, [-1, n_components, output_dim])

        sigmas = tf.clip_by_value(tf.exp(log_sigmas), 1e-5, 1e5)

        y_true_exp = tf.expand_dims(y_true, axis=1)

        exponent = -0.5 * tf.reduce_sum(
            ((y_true_exp - means) / sigmas) ** 2 +
            2 * tf.math.log(sigmas) +
            tf.math.log(2 * math.pi),
            axis=-1
        )

        log_weights = tf.nn.log_softmax(logits, axis=-1)
        log_prob = tf.reduce_logsumexp(log_weights + exponent, axis=-1)

        return -tf.reduce_mean(log_prob)

    return loss_fn


# =========================
# MDN Mode Inference
# =========================
def get_mode_from_mdn(y_pred, n_components, output_dim):
    logits = y_pred[:, :n_components]
    means = y_pred[:, n_components:n_components + n_components * output_dim]
    means = means.reshape(-1, n_components, output_dim)

    best_components = np.argmax(logits, axis=-1)
    modes = np.array([means[i, best_components[i]] for i in range(len(best_components))])
    return modes


# =========================
# Generator Class
# =========================
class PairGenerator(Sequence):
    """
    Keras Sequence generator for MDN training.
    Yields batches of (y_input, x_target).
    """
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True):
        self.x = x_data
        self.y = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.y[batch_idx], self.x[batch_idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================
# Main
# =========================
if __name__ == "__main__":
    collect_NN_mse = []

    # Sample data and compute ground truth
    X = sample_data(num_data_points)
    Y = ground_truth(X)  # No multiplicative noise

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split(
        X, Y, val_pct=0.2, test_pct=0.2, seed=0
    )

    n_components = 50

    output_dim = x_train.shape[-1] if len(x_train.shape) > 1 else 1
    input_dim = y_train.shape[-1] if len(y_train.shape) > 1 else 1

    if len(x_train.shape) == 1:
        x_train, x_val, x_test = x_train.reshape(-1, 1), x_val.reshape(-1, 1), x_test.reshape(-1, 1)
    if len(y_train.shape) == 1:
        y_train, y_val, y_test = y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1)

    # Only create generator for training
    train_gen = PairGenerator(x_train, y_train, batch_size=32)

    for i in range(5):
        np.random.seed(i)
        tf.keras.utils.set_random_seed(i)

        mdn_output_size = n_components + n_components * output_dim + n_components * output_dim

        model = keras.Sequential([
            layers.Dense(640, activation="relu", input_shape=(input_dim,)),
            layers.Dense(640, activation="relu"),
            layers.Dense(mdn_output_size)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=mdn_loss(n_components, output_dim)
        )

        early_stopper = EarlyStopping(
            monitor='loss',
            patience=200,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=40
        )

        # Train using training generator, but pass validation arrays directly
        history = model.fit(
            train_gen,
            validation_data=(y_val, x_val),
            epochs=1000,
            callbacks=[early_stopper,reduce_lr],
            verbose=0
        )

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Negative Log Likelihood")
        plt.legend(["Train", "Val"])
        plt.title(f"MDN Training Curve - Run {i+1}")
        plt.show()

        y_pred_params = model.predict(y_test)

        x_preds = get_mode_from_mdn(
            y_pred_params,
            n_components,
            output_dim
        )

        if x_preds.shape[-1] == 1:
            x_preds = x_preds.flatten()

        y_reconstructed = ground_truth(
            enforce_boundaries(x_preds)
        )

        y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
        y_reconstructed_flat = y_reconstructed.flatten() if len(y_reconstructed.shape) > 1 else y_reconstructed

        NN_mse = np.mean((y_reconstructed_flat - y_test_flat) ** 2)

        print(f"  MDN MSE = {NN_mse:.6f}")
        collect_NN_mse.append(NN_mse)

    np.savez(
        "mdn_results_with_generator_stable_val.npz",
        NN_mse=np.array(collect_NN_mse)
    )
