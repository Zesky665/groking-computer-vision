import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras.models import Model

from deeplearning_models import functional_model, MyModel
from my_utils import display_some_examples



# tensorflow.keras.Sequential
# Sequential model
# model = tf.keras.Sequential()
seq_model = tf.keras.models.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
        MaxPool2D(pool_size=(2,2)),
        BatchNormalization(),
        
        Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
        MaxPool2D(),
        BatchNormalization(),
        
        GlobalAvgPool2D(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ]
)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
   
    print("x_train.shape: ", x_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_test.shape: ", y_test.shape)
   
    if False:
        display_some_examples(x_train, y_train)
        print("x_train.shape: ", x_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("x_test.shape: ", x_test.shape)
        print("y_test.shape: ", y_test.shape)
    

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    model = functional_model()
    # Model settings
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Model training
    model.fit(x_train, y_train, epochs=10, validation_split = 0.2, batch_size=32)
    
    # Model evaluation
    model.evaluate(x_test, y_test, batch_size=32)