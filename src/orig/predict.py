from glob import glob
import os

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tqdm import tqdm

import input_data
import models

def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture):
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)
  fingerprint_input = contrib_audio.mfcc(
      spectrogram,
      decoded_sample_data.sample_rate,
      dct_coefficient_count=dct_coefficient_count)
  fingerprint_frequency_size = model_settings['dct_coefficient_count']
  fingerprint_time_size = model_settings['spectrogram_length']
  reshaped_input = tf.reshape(fingerprint_input, [
      -1, fingerprint_time_size * fingerprint_frequency_size
  ])

  logits = models.create_model(
      reshaped_input, model_settings, model_architecture, is_training=False,
      runtime_settings=runtime_settings)
  prediction_node = tf.argmax(logits, axis=-1)
  return wav_data_placeholder, prediction_node

  # Create an output to use for inference.
  # tf.nn.softmax(logits, name='labels_softmax')

def run(params, tests, labels, wanted_words_original):
  sess = tf.InteractiveSession()
  wav_data_placeholder, prediction_node = create_inference_graph(params['wanted_words'], params['sample_rate'],
                         params['clip_duration_ms'], params['clip_stride_ms'],
                         params['window_size_ms'], params['window_stride_ms'],
                         params['dct_coefficient_count'], params['model_architecture'])
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, tf.train.latest_checkpoint('./model'))

  result = []
  for test in tqdm(tests):
      prediction = getSinglePrediction(test['filepath'], wav_data_placeholder, prediction_node, sess)
      label = labels[prediction[0]]
      label = getOriginalLabel(label, wanted_words_original)
      result.append({ 'filename': test['filename'], 'label': label })
  return result

def getOriginalLabel(label, wanted_words_original):
    if label == 'silence' or label == 'unknown':
        return label
    if label in wanted_words_original:
        return label
    return 'unknown'

def getSinglePrediction(filename, input_node, prediction_node, sess):
    with open(filename, 'rb') as wav_file:
      wav_data = wav_file.read()
    return sess.run(prediction_node, feed_dict={ input_node:  wav_data })

def getTest(filepath):
    return { 'filepath': filepath, 'filename': os.path.basename(filepath) }

def load_labels(filename):
  return [line.rstrip().replace('_', '') for line in tf.gfile.GFile(filename)]

params=dict(
    wanted_words='bed,bird,cat,dog,down,eight,five,four,go,house,left,nine,no,off,on,one,right,seven,six,stop,three,tree,two,up,wow,yes,zero',
    sample_rate=16000,
    clip_duration_ms=1000,
    clip_stride_ms=30,
    window_size_ms=30,
    window_stride_ms=10,
    dct_coefficient_count=40,
    model_architecture='conv',
)
labels=load_labels('./model/conv_labels.txt')
test_data_paths = glob('../../data/test/audio/*wav')
tests = [ getTest(x) for x in test_data_paths ]
# tests = tests[:3000]
wanted_words_original=set('yes,no,up,down,left,right,on,off,stop,go'.split(','))
results = run(params, tests, labels, wanted_words_original)

with open('./model/submission.csv', 'w') as fout:
    print('Start writing file')
    fout.write('fname,label\n')
    for result in results:
        fout.write('{},{}\n'.format(result['filename'], result['label']))
