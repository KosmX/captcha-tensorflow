import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models

from ai_config import *

def new_model():
    input_layer = tf.keras.Input(shape=(H, W, C))
    x = layers.Conv2D(32, 3, activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)

    x = layers.Dense(D * N_LABELS, activation='softmax')(x)
    x = layers.Reshape((D, N_LABELS))(x)

    model = models.Model(inputs=input_layer, outputs=x)

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics= ['accuracy'])
    model.summary()
    return(model)

def load_model(target):
    model = tf.keras.models.load_model(target)
    model.summary()
    return(model)

    # include 1 model
