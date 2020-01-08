import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def wiggle_function(t, n0, gammatau, A, omega_a, phi):
    return( ( n0 / gammatau ) * np.exp(-t / gammatau) * (1 - A * np.cos( omega_a * t + phi) ) )
def wiggle_tf(t, n0, gammatau, A, omega_a, phi):
    y_n = n0
    y_e = tf.math.exp( tf.math.divide(tf.math.negative(t) , gammatau) )
    #y_e = tf.math.exp( tf.math.divide(-t , gammatau) )
    y_c = tf.math.subtract( tf.Variable(1.0) , tf.math.multiply(A , tf.math.cos(tf.math.add ( tf.math.multiply( omega_a , t), phi ) ) ) )
    y_pred = tf.math.multiply( y_c , tf.math.multiply( y_n , y_e) )
    return y_pred

