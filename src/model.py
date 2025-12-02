# src/model.py
"""
实现 build_mlp()（严格复现 notebook 中的 MLP）以及一个可选的简单 CNN（用于后续对比）
使用 TensorFlow Keras 构建
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer, Reshape, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization, Activation, Add
from config import Config

def build_mlp(img_shape=None, num_classes=None):
    """
    MLP that accepts image-shaped input and flattens it internally.
    Compatible with dataset pipeline returning (H, W, C) tensors.
    """
    img_shape = img_shape or (Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)
    num_classes = num_classes or Config.NUM_CLASSES
    model = Sequential(name="MLP_Classifier")
    model.add(InputLayer(input_shape=img_shape))   # e.g., (64, 64, 1)
    model.add(Flatten())                           # ←← 关键：展平为 (4096,)
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


def build_mlp_v1_gelu_ln(input_shape, num_classes):
    """
    MLP with GELU activation and Layer Normalization
    Reference: Hendrycks & Gimpel (2016); Ba et al. (2016)
    """
    model = Sequential(name="MLP_v1_GELU_LN")
    model.add(InputLayer(input_shape=input_shape))
    
    # If input is 3D, add Flatten layer
    if len(input_shape) == 3:
        model.add(Flatten())
    
    # Hidden layers with LayerNorm and GELU
    model.add(Dense(1024, name='fc1'))
    model.add(LayerNormalization(name='ln1'))
    model.add(Activation('gelu', name='gelu1'))
    model.add(Dropout(0.5, name='dropout1'))
    
    model.add(Dense(512, name='fc2'))
    model.add(LayerNormalization(name='ln2'))
    model.add(Activation('gelu', name='gelu2'))
    model.add(Dropout(0.4, name='dropout2'))
    
    model.add(Dense(256, name='fc3'))
    model.add(LayerNormalization(name='ln3'))
    model.add(Activation('gelu', name='gelu3'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='predictions'))
    
    return model


def build_mlp_v2_residual(input_shape, num_classes):
    """
    MLP with residual connections, GELU activation, and Layer Normalization
    Reference: He et al. (2016); Tolstikhin et al. (2021)
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input')
    
    # Initial flatten if input is 3D
    if len(input_shape) == 3:
        x = Flatten(name='flatten')(inputs)
    else:
        x = inputs
    
    # Block 1: 1024 → 1024 with residual connection
    shortcut = Dense(1024, name='shortcut1')(x)
    x = Dense(1024, name='fc1_1')(x)
    x = LayerNormalization(name='ln1_1')(x)
    x = Activation('gelu', name='gelu1_1')(x)
    x = Dropout(0.5, name='dropout1_1')(x)
    x = Dense(1024, name='fc1_2')(x)
    x = Add(name='add1')([shortcut, x])
    
    # Block 2: 1024 → 512 with residual connection
    shortcut = Dense(512, name='shortcut2')(x)
    x = Dense(512, name='fc2_1')(x)
    x = LayerNormalization(name='ln2_1')(x)
    x = Activation('gelu', name='gelu2_1')(x)
    x = Dropout(0.4, name='dropout2_1')(x)
    x = Dense(512, name='fc2_2')(x)
    x = Add(name='add2')([shortcut, x])
    
    # Block 3: 512 → 256 without residual connection
    x = Dense(256, name='fc3')(x)
    x = LayerNormalization(name='ln3')(x)
    x = Activation('gelu', name='gelu3')(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="MLP_v2_Residual")
    
    return model


def build_mlp_ablation_gelu_only(input_shape, num_classes):
    """
    MLP Ablation Study: GELU Only
    Reference: Hendrycks & Gimpel (2016)
    """
    model = Sequential(name="MLP_Ablation_GELU_Only")
    model.add(InputLayer(input_shape=input_shape))
    
    # If input is 3D, add Flatten layer
    if len(input_shape) == 3:
        model.add(Flatten())
    
    # Hidden layers with GELU activation but no LayerNorm
    model.add(Dense(1024, name='fc1'))
    model.add(Activation('gelu', name='gelu1'))
    model.add(Dropout(0.5, name='dropout1'))
    
    model.add(Dense(512, name='fc2'))
    model.add(Activation('gelu', name='gelu2'))
    model.add(Dropout(0.4, name='dropout2'))
    
    model.add(Dense(256, name='fc3'))
    model.add(Activation('gelu', name='gelu3'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='predictions'))
    
    return model


def build_mlp_ablation_layernorm_only(input_shape, num_classes):
    """
    MLP Ablation Study: LayerNorm Only
    Reference: Ba et al. (2016)
    """
    model = Sequential(name="MLP_Ablation_LayerNorm_Only")
    model.add(InputLayer(input_shape=input_shape))
    
    # If input is 3D, add Flatten layer
    if len(input_shape) == 3:
        model.add(Flatten())
    
    # Hidden layers with LayerNorm but ReLU activation
    model.add(Dense(1024, name='fc1'))
    model.add(LayerNormalization(name='ln1'))
    model.add(Activation('relu', name='relu1'))
    model.add(Dropout(0.5, name='dropout1'))
    
    model.add(Dense(512, name='fc2'))
    model.add(LayerNormalization(name='ln2'))
    model.add(Activation('relu', name='relu2'))
    model.add(Dropout(0.4, name='dropout2'))
    
    model.add(Dense(256, name='fc3'))
    model.add(LayerNormalization(name='ln3'))
    model.add(Activation('relu', name='relu3'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='predictions'))
    
    return model


