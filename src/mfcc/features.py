import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

class FeatureGenerator(object):
    def __init__(self, model_settings, bgFileNames=None):
        self.sess = tf.Session()
        self.model_settings = model_settings
        self._prepare_processing_graph(model_settings)
        self._prepare_audio_reading_graph()
        self._prepare_bg_data(bgFileNames)

    def getFeatures(self, filename, fg_vol=1, bg_vol=0):
        desired_samples = self.model_settings['desired_samples']
        bg_data = self.getRandomBgData(desired_samples)
        feed_dict = {
            self.wav_filename_placeholder_: filename,
            self.time_shift_padding_placeholder_: [[0, 0], [0, 0]],
            self.time_shift_offset_placeholder_: [0, 0],
            self.background_data_placeholder_: bg_data,
            self.background_volume_placeholder_: bg_vol,
            self.foreground_volume_placeholder_: fg_vol,
        }
        # Run the graph to produce the output audio.
        return self.sess.run(self.mfcc_, feed_dict=feed_dict).flatten()

    def getAudio(self, filename):
        feed_dict = { self.bg_wav_filename_placeholder_: filename }
        return self.sess.run(self.bg_wav_decoder_, feed_dict=feed_dict).audio.flatten()

    def getRandomBgData(self, desired_samples):
        if len(self.background_data) == 0:
            return np.zeros([desired_samples, 1])
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        background_offset = np.random.randint(0, len(background_samples) - desired_samples)
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        return background_clipped.reshape([desired_samples, 1])

    def _prepare_bg_data(self, bgFileNames):
        self.background_data = []
        if bgFileNames:
            self.background_data = [ self.getAudio(x) for x in bgFileNames ]

    def _prepare_processing_graph(self, model_settings):
        desired_samples = model_settings['desired_samples']

        # Set input placeholders
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        # Mix in background noise.
        self.background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])

        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
        scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)
        padded_foreground = tf.pad(scaled_foreground, self.time_shift_padding_placeholder_, mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, [desired_samples, -1])
        background_mul = tf.multiply(self.background_data_placeholder_, self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)

        # Set output operation
        self.mfcc_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['dct_coefficient_count'])

    def _prepare_audio_reading_graph(self):
        # placeholder
        self.bg_wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.bg_wav_filename_placeholder_)
        # set output operation
        self.bg_wav_decoder_ = contrib_audio.decode_wav(wav_loader, desired_channels=1)

def main():
    model_settings={
        'desired_samples': 16000, 
        'window_size_samples': 480, 
        'window_stride_samples': 160, 
        'spectrogram_length': 98, 
        'dct_coefficient_count': 40, 
        'fingerprint_size': 3920, 
        'label_count': 12, 
        'sample_rate': 16000
    }
    bgFileNames = [
            '../../data/train/audio/_background_noise_/doing_the_dishes.wav', 
            '../../data/train/audio/_background_noise_/dude_miaowing.wav'
    ]
    featureGenerator = FeatureGenerator(model_settings, bgFileNames)
    x = featureGenerator.getFeatures('zero', '../../data/train/audio/zero/e9287461_nohash_1.wav')
    y = featureGenerator.getAudio('../../data/train/audio/_background_noise_/running_tap.wav')
    print('x: ', x.shape)
    print('y: ', y.shape)
    print(featureGenerator.background_data)
    print(featureGenerator.background_data[0].shape)
    print(featureGenerator.background_data[1].shape)
    print('bg data: ', featureGenerator.getRandomBgData(16000))
    print('bg data: ', featureGenerator.getRandomBgData(16000).shape)

if __name__ == "__main__":
    main()
