import os
import json
import tensorflow as tf
from config import Config
from utils import ensure_dir

def train_model(model, train_ds, test_ds, model_name="model"):
    ensure_dir(Config.LOG_DIR)
    ensure_dir(Config.CKPT_DIR)
    ensure_dir(Config.FIGURE_DIR)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=Config.EPOCHS
    )

    # 保存训练日志
    log_path = os.path.join(Config.LOG_DIR, f"{model_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(history.history, f)

    # 保存模型
    model_ckpt_dir = os.path.join(Config.CKPT_DIR, model_name)
    ensure_dir(model_ckpt_dir)
    ckpt_path = os.path.join(model_ckpt_dir, "model.keras")
    model.save(ckpt_path)

    print(f"训练日志已保存到: {log_path}")
    print(f"模型已保存到: {ckpt_path}")

    return history 