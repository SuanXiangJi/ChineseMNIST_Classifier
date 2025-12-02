import os

class Config:
    # 数据集路径（适配你的目录）
    TRAIN_DIR = "dataset/train"
    TEST_DIR = "dataset/test"

    LOG_DIR = "experiments/logs"
    CKPT_DIR = "experiments/checkpoints"
    FIGURE_DIR = "experiments/figures"

    IMG_SIZE = (64, 64)
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_CHANNELS = 1
    NUM_CLASSES = 15

    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-3
