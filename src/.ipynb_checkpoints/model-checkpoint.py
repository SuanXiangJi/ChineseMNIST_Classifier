# src/model.py
"""
实现 build_mlp()（严格复现 notebook 中的 MLP）以及一个可选的简单 CNN（用于后续对比）
使用 TensorFlow Keras 构建
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from config import Config

def build_mlp(input_dim=None, num_classes=None):
    """
    Notebook-style MLP:
      Flatten -> Dense(1024, relu) -> Dense(512, relu) -> Dense(256, relu) -> Dense(num_classes, softmax)
    input_dim: 整数，输入维度（H*W*1）
    """
    input_dim = input_dim or (Config.IMG_HEIGHT * Config.IMG_WIDTH * Config.IMG_CHANNELS)
    num_classes = num_classes or Config.NUM_CLASSES

    model = Sequential(name="MLP_Classifier")
    model.add(InputLayer(input_shape=(input_dim,)))
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dropout(0.4, name='dropout2'))
    model.add(Dense(256, activation='relu', name='fc3'))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))
    return model

def build_cnn(img_shape=None, num_classes=None):
    """
    简单 CNN baseline（可选）
    """
    img_shape = img_shape or (Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)
    num_classes = num_classes or Config.NUM_CLASSES

    model = Sequential(name="CNN_Classifier")
    model.add(InputLayer(input_shape=img_shape))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model
