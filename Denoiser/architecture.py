import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(64, 3, padding="same")
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(64, 3, padding="same")

    def call(self, x):
        skip = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + skip


class RCNN(tf.keras.Model):
    def __init__(self, num_blocks=4):
        super().__init__()
        self.head = layers.Conv2D(64, 3, padding="same")
        self.relu = layers.ReLU()
        self.blocks = [ResidualBlock() for _ in range(num_blocks)]
        self.tail = layers.Conv2D(3, 2, padding="same")

    def call(self, x):
        x = self.relu(self.head(x))
        for b in self.blocks:
            x = b(x)
        return self.tail(x)
