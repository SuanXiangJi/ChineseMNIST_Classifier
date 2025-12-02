#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成消融实验结果的对比图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 配置路径
LOG_DIR = "experiments/logs"
CHECKPOINT_DIR = "experiments/checkpoints"
FIGURE_DIR = "experiments/figures"
OUTPUT_DIR = "experiments/tables"

# 确保输出目录存在
os.makedirs(FIGURE_DIR, exist_ok=True)

# 设置绘图样式
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})


# 消融实验模型配置
ABLATION_MODELS = {
    "mlp": {
        "name": "MLP_Baseline",
        "color": "red",
        "linestyle": "-",
        "linewidth": 1.5
    },
    "mlp_v1": {
        "name": "MLP-v1 (GELU+LN)",
        "color": "orange",
        "linestyle": "-",
        "linewidth": 2.0
    },
    "mlp_ablation_gelu_only": {
        "name": "Ablation (GELU Only)",
        "color": "green",
        "linestyle": "--",
        "linewidth": 1.5
    },
    "mlp_ablation_layernorm_only": {
        "name": "Ablation (LN Only)",
        "color": "brown",
        "linestyle": "--",
        "linewidth": 1.5
    }
}


def load_log_data(model_name):
    """加载指定模型的日志数据"""
    log_path = os.path.join(LOG_DIR, f"{model_name}_log.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Log file not found: {log_path}")
        return None


def plot_ablation_convergence():
    """
    绘制消融实验的收敛曲线对比
    """
    plt.figure(figsize=(12, 6))
    
    for model_key, config in ABLATION_MODELS.items():
        log_data = load_log_data(model_key)
        if log_data:
            # 获取训练和验证准确率
            train_acc = log_data["accuracy"]
            val_acc = log_data["val_accuracy"]
            epochs = range(1, len(train_acc) + 1)
            
            # 绘制训练准确率
            plt.plot(epochs, train_acc, 
                     linestyle='--',
                     color=config["color"],
                     linewidth=config["linewidth"],
                     label=f"{config['name']} (Train)")
            
            # 绘制验证准确率
            plt.plot(epochs, val_acc, 
                     linestyle='-',
                     color=config["color"],
                     linewidth=config["linewidth"],
                     label=f"{config['name']} (Val)")
    
    # 设置图表属性
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Ablation Study: Convergence Comparison")
    plt.xlim(0, 20)
    plt.ylim(0.6, 1.0)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_convergence.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_convergence.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Ablation convergence plot saved successfully.")


def plot_ablation_final_accuracy():
    """
    绘制消融实验的最终准确率对比柱状图
    """
    # 收集最终准确率数据
    final_accuracies = {}
    
    for model_key, config in ABLATION_MODELS.items():
        log_data = load_log_data(model_key)
        if log_data:
            final_val_acc = log_data["val_accuracy"][-1] * 100  # 转换为百分比
            final_accuracies[config["name"]] = {
                "accuracy": final_val_acc,
                "color": config["color"]
            }
    
    # 按准确率排序
    sorted_models = sorted(final_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    plt.figure(figsize=(10, 6))
    
    # 提取数据
    model_names = [name for name, _ in sorted_models]
    accuracies = [data["accuracy"] for _, data in sorted_models]
    colors = [data["color"] for _, data in sorted_models]
    
    # 绘制柱状图
    bars = plt.bar(model_names, accuracies, color=colors, edgecolor="black", linewidth=1)
    
    # 添加数值标签
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{accuracy:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 设置图表属性
    plt.xlabel("Model")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Ablation Study: Final Accuracy Comparison")
    plt.ylim(75, 90)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_final_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_final_accuracy.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Ablation final accuracy plot saved successfully.")


def plot_ablation_overfitting_gap():
    """
    绘制消融实验的过拟合gap对比
    """
    plt.figure(figsize=(12, 6))
    
    for model_key, config in ABLATION_MODELS.items():
        log_data = load_log_data(model_key)
        if log_data:
            # 计算每epoch的过拟合gap
            train_acc = log_data["accuracy"]
            val_acc = log_data["val_accuracy"]
            gap = [train_acc[i] - val_acc[i] for i in range(len(train_acc))]
            epochs = range(1, len(gap) + 1)
            
            # 绘制gap曲线
            plt.plot(epochs, gap,
                     linestyle=config["linestyle"],
                     color=config["color"],
                     linewidth=config["linewidth"],
                     label=config["name"])
    
    # 设置图表属性
    plt.xlabel("Epoch")
    plt.ylabel("Overfitting Gap")
    plt.title("Ablation Study: Overfitting Dynamics")
    plt.xlim(0, 20)
    plt.ylim(0.0, 0.15)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_overfitting_gap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURE_DIR, "ablation_overfitting_gap.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Ablation overfitting gap plot saved successfully.")


def main():
    """主函数"""
    print("Generating ablation study plots...")
    
    # 生成消融实验收敛曲线对比
    plot_ablation_convergence()
    
    # 生成消融实验最终准确率对比柱状图
    plot_ablation_final_accuracy()
    
    # 生成消融实验过拟合gap对比
    plot_ablation_overfitting_gap()
    
    print("All ablation study plots generated successfully!")


if __name__ == "__main__":
    main()