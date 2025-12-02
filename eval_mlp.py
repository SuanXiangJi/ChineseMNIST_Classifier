# eval_mlp.py
# 评估 MLP 模型

# 可选：强制使用 CPU（调试时取消注释）
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os
import tensorflow as tf
from src.dataset import load_datasets
from src.evaluator import evaluate_model
from config import Config


def main():
    print("Loading datasets...")
    train_ds, test_ds = load_datasets()  # 返回 (B, 64, 64, 1) + one-hot labels

    print("Loading MLP model...")
    # 加载训练好的模型
    model_path = os.path.join(Config.CKPT_DIR, "mlp", "model.keras")
    if not os.path.exists(model_path):
        # 兼容旧的模型路径
        model_path = os.path.join(Config.CKPT_DIR, "model_mlp_final.h5")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
    
    model = tf.keras.models.load_model(model_path)
    model.summary()

    print("Starting evaluation...")
    acc, cm, report = evaluate_model(model, test_ds)

    print("MLP evaluation completed.")


if __name__ == "__main__":
    main()