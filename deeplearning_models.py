import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras.models import Model
import numpy as np

# functional approach :  function that returns a model
def functional_model():
    inputs = Input(shape=(28,28,1))
    
    x = Conv2D(filters=32, kernel_size=(3,3), activation="relu")(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=128, kernel_size=(3,3), activation="relu")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)
    
    return Model(inputs, x)

# tensorflow.keras.Model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        self.conv2 = Conv2D(filters=64, kernel_size=(3,3), activation="relu")
        self.maxpool1 = MaxPool2D(pool_size=(2,2))
        self.batchnorm1 = BatchNormalization()
        
        self.conv3 = Conv2D(filters=128, kernel_size=(3,3), activation="relu")
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()
        
        self.globalavgpool = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(10, activation="softmax")
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        
        x = self.globalavgpool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x

