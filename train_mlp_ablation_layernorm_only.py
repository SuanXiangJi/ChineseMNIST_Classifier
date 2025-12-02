# train_mlp_ablation_layernorm_only.py
# 训练 MLP 消融实验模型：仅使用 LayerNorm

# 可选：强制使用 CPU（调试时取消注释）
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.dataset import load_datasets
from src.model import build_mlp_ablation_layernorm_only
from src.trainer import train_model
from config import Config


if __name__ == "__main__":
    print("Loading datasets...")
    train_ds, test_ds = load_datasets()  # 返回 (B, 64, 64, 1) + one-hot labels

    print("Building MLP Ablation (LayerNorm Only) model...")
    # 注意：build_mlp_ablation_layernorm_only 接受原始图像 shape，并内部展平
    model = build_mlp_ablation_layernorm_only(
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS),
        num_classes=Config.NUM_CLASSES
    )
    model.summary()

    print("Starting training...")
    train_model(model, train_ds, test_ds, model_name="mlp_ablation_layernorm_only")

    print("MLP Ablation (LayerNorm Only) training completed.")