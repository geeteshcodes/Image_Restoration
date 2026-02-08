from .architecture import Colorizer_AutoEncoder
import tensorflow as tf
WEIGHTS = "Colour/colourizer_weights.h5"

def load_model():
    model = Colorizer_AutoEncoder()
    dummy=tf.zeros((1,128,128,1))
    model(dummy)
    model.load_weights(WEIGHTS)
    return model

