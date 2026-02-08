import tensorflow as tf
from tensorflow.keras import layers

class Colorizer_AutoEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # -------- Encoder --------
        self.e1 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.e2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.e3 = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')
        # -------- Bottleneck --------
        self.b1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.drop = layers.Dropout(0.2)
        self.b2 = layers.Conv2D(256, 3, padding='same', activation='relu')

        # -------- Decoder --------
        
        self.d1 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.u1 = layers.UpSampling2D(2)
        self.r1 = layers.Conv2D(128, 1, activation='relu')

        self.d2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.u2 = layers.UpSampling2D(2)
        self.r2 = layers.Conv2D(64, 1, activation='relu')

        self.d3 = layers.Conv2D(32,3, padding='same', activation='relu')
        self.u3 = layers.UpSampling2D(2)
        self.r3 = layers.Conv2D(32, 1, activation='relu')

        # -------- Output (ab channels) --------
        self.out = layers.Conv2D(2, 1, activation='tanh')

    # -------- Forward pass --------
    def call(self, L):
        e1 = self.e1(L)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        x = self.b1(e3)
        x=self.drop(x)
        x = self.b2(x)
        x = self.d1(x)
        x = self.r1(tf.concat([x, e3], axis=-1))
        x = self.u1(x)

        x = self.d2(x)
        x = self.r2(tf.concat([x, e2], axis=-1))
        x = self.u2(x)

        x = self.d3(x)
        x = self.r3(tf.concat([x, e1], axis=-1))
        x = self.u3(x)

        return self.out(x)


