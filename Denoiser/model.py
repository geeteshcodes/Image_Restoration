from .architecture import RCNN
import tensorflow as tf
WEIGHTS = "denoiser/denoiser_weights.h5"

def load_model():
    model = RCNN()
    dummy=tf.zeros((1,128,128,3))
    
    model(dummy)
    model.load_weights(WEIGHTS)
    return model
