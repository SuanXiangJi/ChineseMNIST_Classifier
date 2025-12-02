# src/dataset.py
import tensorflow as tf
from config import Config

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",          # ←←← 关键！强制加载为单通道
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        seed=42
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Config.TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",          # ←←← 同样这里也要加
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        seed=42
    )

    return train_ds, test_ds