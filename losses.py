# Copyright 2024 Google LLC.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Losses functions to be used be model training."""

from typing import Any, cast

import keras.backend as K  # noqa: N812
import numpy as np
import tensorflow as tf


def masked_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Custom MSE loss function that ignores pixels marked as irrelevant."""
    weights = y_true[..., -1:]
    mask = K.greater(weights, 0)  # No need for casting

    y_true_values = y_true[..., :-1]

    return K.mean(tf.where(mask, K.square(y_pred - y_true_values), 0.0))


def density_combined_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.5, mse_scale: float = 100.0
) -> tf.Tensor:
    """Combined loss for density.

    Merges MSE and tree count rate loss.
    """
    mse_loss = masked_mse(y_true, y_pred) * mse_scale
    count_loss = tree_count_rate_loss(y_true, y_pred)
    return alpha * mse_loss + (1 - alpha) * count_loss


def tree_count_rate_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """A loss function that cares about predicted number of trees."""
    weights = y_true[..., -1:]
    mask = K.greater(weights, 0)

    y_true_values = y_true[..., :-1]

    true_count = tf.reduce_sum(tf.where(mask, y_true_values, 0.0))
    pred_count = tf.reduce_sum(tf.where(mask, y_pred, 0.0))

    return K.square(1.0 - (pred_count / (true_count + K.epsilon())))


def tree_count_rate(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute accuracy in terms of tree count."""
    # Extract the relevance mask (last channel of y_true)
    weights = y_true[..., -1:]
    mask = K.cast(K.greater(weights, 0), K.floatx())

    y_true_values = y_true[..., :-1]

    true_count = tf.reduce_sum(mask * y_true_values)
    pred_count = tf.reduce_sum(mask * y_pred)
    return pred_count / (true_count + 0.0001)


def tversky(
    y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.6, beta: float = 0.4
) -> tf.Tensor:
    """Function to calculate the Tversky loss for imbalanced data.

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    # weights
    y_weights = y_true[..., 1]
    y_weights = y_weights[..., np.newaxis]

    ones = 1
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    EPSILON = 0.00001  # noqa: N806
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)


def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute accuracy."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))


def dice_coef(
    y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 0.0000001
) -> tf.Tensor:
    """Compute dice coefficient."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]

    y_t = np.squeeze(K.flatten(y_t))
    y_pred = np.squeeze(K.flatten(y_pred).numpy())
    intersection = K.sum(K.abs(y_t * y_pred))
    union = K.sum(y_t) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute dice loss."""
    return 1 - dice_coef(y_true, y_pred)


def true_positives(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute true positive."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round(y_t * y_pred)


def false_positives(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute false positive."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round((1 - y_t) * y_pred)


def true_negatives(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute true negative."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))


def false_negatives(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute false negative."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round((y_t) * (1 - y_pred))


def sensitivity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute sensitivity (recall)."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))


def specificity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute specificity (precision)."""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))


def miou(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Compute Mean Intersection over Union (IoU).

    :param y_true: ground truth mask tensor
    :param y_pred: predicted mask tensor
    :return: intersection over union ratio
    """
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    mioufuc = tf.keras.metrics.MeanIoU(num_classes=2)
    mioufuc.update_state(y_t, y_pred)
    return cast(float, mioufuc.result().numpy())


def weight_miou(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Compute weighted Mean Intersection over Union (IoU).

    :param y_true: ground truth mask tensor
    :param y_pred: predicted mask tensor
    :return: intersection over union ratio
    """
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    y_weights = y_true[..., 1]
    y_weights = y_weights[..., np.newaxis]
    mioufuc = tf.keras.metrics.MeanIoU(num_classes=2)
    mioufuc.update_state(y_t, y_pred, sample_weight=y_weights)
    return cast(float, mioufuc.result().numpy())


def w_mae(
    y_true: tf.Tensor, y_pred: tf.Tensor, clip_delta: int = 10, wei: int = 5
) -> Any:  #  noqa
    """Loss function for CHM prediction."""
    weights = y_true[..., -1:]
    mask = K.greater(weights, 0)  # No need for casting

    y_true_values = y_true[..., :-1]

    # weight higher for large heights
    cond = y_true_values < clip_delta
    # no spatial weights
    loss_small = K.mean(tf.where(mask, K.abs(y_pred - y_true_values), 0.0))
    loss_large = K.mean(tf.where(mask, wei * K.abs(y_pred - y_true_values), 0.0))
    return tf.where(cond, loss_small, loss_large)


def weighted_mae_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> Any:  #  noqa
    """Loss function for CHM with one value per sample in the batch."""
    # Extract labels and weights from y_true
    labels = y_true[:, :, :, 0]
    weights = y_true[:, :, :, 1]

    # Calculate absolute errors
    absolute_errors = tf.abs(labels - y_pred[:, :, :, 0])

    # Apply weight masking
    weighted_errors = absolute_errors * weights

    # Replace NaNs with zeros
    weighted_errors = tf.where(
        tf.math.is_nan(weighted_errors), tf.zeros_like(weighted_errors), weighted_errors
    )

    # Calculate mean of weighted errors across
    # relevant dimensions (exclude batch dimension)
    return tf.reduce_mean(weighted_errors, axis=[1, 2])


def weighted_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> Any:  #  noqa
    """Loss function for CHM with one value per sample in the batch."""
    # Extract labels and weights from y_true
    labels = y_true[:, :, :, 0]
    weights = y_true[:, :, :, 1]

    # Calculate squared errors
    squared_errors = tf.square(labels - y_pred[:, :, :, 0])

    # Apply weight masking
    weighted_errors = squared_errors * weights

    # Replace NaNs with zeros
    weighted_errors = tf.where(
        tf.math.is_nan(weighted_errors), tf.zeros_like(weighted_errors), weighted_errors
    )

    # Calculate mean of weighted errors across
    # relevant dimensions (exclude batch dimension)
    return tf.reduce_mean(weighted_errors, axis=[1, 2])


def weighted_huber_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 0.03
) -> Any:  # noqa
    """Huber loss function for CHM with one value per sample in the batch."""
    # Extract labels and weights from y_true
    labels = y_true[:, :, :, 0]
    weights = y_true[:, :, :, 1]

    # Calculate absolute errors
    absolute_errors = tf.abs(labels - y_pred[:, :, :, 0])

    # Calculate Huber loss error (either mse or "deltified" mae depending on abs error)
    huber_errors = tf.where(
        absolute_errors <= delta,
        0.5 * tf.square(absolute_errors),
        delta * (absolute_errors - 0.5 * delta),
    )

    # Apply weight masking
    weighted_errors = huber_errors * weights

    # Replace NaNs with zeros
    weighted_errors = tf.where(
        tf.math.is_nan(weighted_errors), tf.zeros_like(weighted_errors), weighted_errors
    )

    # Calculate mean of weighted Huber loss across
    # relevant dimensions (exclude batch dimension)
    return tf.reduce_mean(weighted_errors, axis=[1, 2])
