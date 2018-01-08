import os
import re
from glob import glob

import numpy as np
from scipy.io import wavfile

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

from python_speech_features import mfcc

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
    spectrogram_length=99, 
    dct_coefficient_count=40, 
    fingerprint_size=3960, 
    label_count=12, 
    sample_rate=16000,
)
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}

def getTransformedAudioLocal(fname, desired_samples, fg_vol, bg_data, bg_vol, clip_min, clip_max, time_shift_samples):
    # read as int
    _, wav = wavfile.read(fname)
    # convert to float
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    # scale by fg_vol
    wav = np.multiply(wav, fg_vol)
    # fit to desired_samples
    if len(wav) < desired_samples:
        wav = np.pad(wav, (0, desired_samples - len(wav)), 'constant')
    else:
        wav = wav[:desired_samples]
    # time shift
    # if time_shift_samples >= 0:
        # wav = np.pad(wav, (time_shift_samples, 0), 'constant')[:desired_samples]
    # else:
        # wav = np.pad(wav, (0, -time_shift_samples), 'constant')[-desired_samples:]
    # add random bg after mult with bg_vol
    if bg_vol > 0:
        noise = np.multiply(getRandomBgData(bg_data, desired_samples), bg_vol)
        wav = np.add(wav, noise)
        # clip by value in end
        wav = np.clip(wav, clip_min, clip_max)
    return wav

def getRandomBgData(bg_data, desired_samples):
    bg_samples = bg_data[np.random.randint(len(bg_data))]
    bg_offset = np.random.randint(0, len(bg_samples) - desired_samples)
    return bg_samples[bg_offset:(bg_offset + desired_samples)]


def getMfcc(audio):
    return mfcc(audio, numcep=40,nfilt=40).flatten()

def testMfcc():
    fname='../../data/train/audio/zero/e9287461_nohash_1.wav'
    _, wav = wavfile.read(fname)
    mfcc1=getMfcc(wav)
    print(mfcc1)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    mfcc2=getMfcc(wav)
    print(mfcc2)

def _model_handler(features, labels, mode, params, config):
    model = tf.make_template(
        'my_template', model_def.create_model,
        create_scope_now_=True,
    )
    x = features[FINGERPRINT_KEY]
    x = tf.to_float(x)
    logits = model(x, params, mode == tf.estimator.ModeKeys.TRAIN)

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
                cm2=(logits, tf.constant(1)),
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
