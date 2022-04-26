import numpy as np
import tensorflow.keras.backend as K

def self_bce(y_true, y_pred):
    r = np.where(y_true == 1., 1., -1.)
    y_pred = K.max(y_pred, 1)
    lng = y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8)

    loss = -1e-3 * r * 0.5 * lng
    return loss
