import tensorflow as tf

with tf.Session() as sess:
    print(tf.get_default_session())
    print(sess.sess_str)
    devices = sess.list_devices();
    print(devices);
    for d in devices:
        print(d)

