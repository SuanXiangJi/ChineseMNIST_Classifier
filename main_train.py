# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 加在最上面！
from src.dataset import load_datasets
from src.model import build_cnn
from src.trainer import train_model

if __name__ == "__main__":
    train_ds, test_ds = load_datasets()
    model = build_cnn()
    train_model(model, train_ds, test_ds)
