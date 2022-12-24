#-- since we are doing a binary classifcation it is justifiable
#-- to use bianry cross entropy loss
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    # TODO: return the binary_cross_entropy loss
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    # TODO: return the binary_cross_entropy_prime. 
    # Note, this is the formula on the bottom in the ppt slides.
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)