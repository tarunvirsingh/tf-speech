import os
import re
from glob import glob

import numpy as np
from scipy.io import wavfile

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

import model_def

tf.logging.set_verbosity(tf.logging.INFO)

model_dir = './model'
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
FINGERPRINT_KEY='fingerprint'
params=dict(
    seed=2018,
    keep_prob=0.5,
    learning_rate=1e-3,
    clip_gradients=15.0,
    use_batch_norm=True,
    num_classes=len(POSSIBLE_LABELS),
    desired_samples=16000, 
    window_size_samples=480, 
    window_stride_samples=160, 
    spectrogram_length=98, 
    dct_coefficient_count=40, 
    fingerprint_size=3920, 
    label_count=12, 
    sample_rate=16000,
)
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}

def _model_handler(features, labels, mode, params, config):
    model = tf.make_template(
        'my_template', model_def.create_model,
        create_scope_now_=True,
    )
    logits = model(features[FINGERPRINT_KEY], params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                # acc=(acc, acc_op),
                acc=tf.metrics.mean_per_class_accuracy(labels, prediction, params.num_classes),
                cm=eval_confusion_matrix(labels, prediction, params.num_classes),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),
            'fname': features['fname'], # it's a hack for simplicity
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)

def eval_confusion_matrix(labels, predictions, num_classes):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions)
        con_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        update_op = tf.assign_add(con_matrix_sum, con_matrix)
        return tf.convert_to_tensor(con_matrix_sum), update_op

def create_estimator(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=_model_handler,
        config=config,
        params=hparams,
    )
