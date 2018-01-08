import tensorflow as tf
from tensorflow.contrib.training import HParams

def create_model(fingerprint_input, params, is_training):
  # if is_training:
    # dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    # dropout_prob = tf.constant(0.5, dtype=tf.float32, name='dropout_prob')
  input_frequency_size = params.dct_coefficient_count
  input_time_size = params.spectrogram_length
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.get_variable('first_weights',
      initializer=tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.get_variable('first_bias', initializer=tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, params.keep_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.get_variable('second_weights',
      initializer=tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.get_variable('second_bias', initializer=tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, params.keep_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = params.label_count
  final_fc_weights = tf.get_variable('final_fc_weights',
      initializer=tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.get_variable('final_fc_bias', initializer=tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  return final_fc
  # if is_training:
    # return final_fc, dropout_prob
  # else:
    # return final_fc, 0.5 # fix this. remove the 0.5

def main():
    fingerprint = tf.zeros([2, 3920])
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
    hparams=HParams(**model_settings)
    is_training = True
    model = create_model(fingerprint, hparams, is_training)
    print(model)

if __name__ == "__main__":
    main()
