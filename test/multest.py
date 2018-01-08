import tensorflow as tf

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    a = tf.random_normal([20000, 3000], mean=0, stddev=1)
    b = tf.random_normal([3000, 3000], mean=0, stddev=1)
    c = tf.reduce_sum(tf.matmul(a, b));
    print(c.eval())

