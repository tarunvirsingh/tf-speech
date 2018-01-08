import tensorflow as tf
import numpy as np
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from scipy.io import wavfile

filename = '/Users/tsingh1/Developer/kaggle/speech/data/train/audio/bed/d78858d9_nohash_1.wav';
filenameTensor = tf.constant(filename);
with tf.Session() as sess:
    wav_loader = io_ops.read_file(filenameTensor)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=5
    )
    x = wav_decoder.audio.eval().flatten()
    print('x1', x)
    print('x1', x.shape)

_, wav = wavfile.read(filename)
wav1 = wav.astype(np.float32) / np.iinfo(np.int16).max
print('w', wav)
print('w1', wav1)
print('w1', wav1.shape)
