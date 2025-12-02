import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 加在最上面！
from tensorflow.keras.models import load_model
from src.dataset import load_datasets
from src.evaluator import evaluate_model
from config import Config

if __name__ == "__main__":
    train_ds, test_ds = load_datasets()
    model = load_model("experiments/checkpoints/model_final.h5")
    evaluate_model(model, test_ds)
