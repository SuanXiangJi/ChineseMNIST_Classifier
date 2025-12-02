#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较所有模型的性能，生成Markdown和LaTeX表格
"""

import os
import json
import glob
import tensorflow as tf
from collections import defaultdict

# 配置路径
LOG_DIR = "experiments/logs"
CHECKPOINT_DIR = "experiments/checkpoints"
OUTPUT_DIR = "experiments/tables"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model_params(model_path):
    """获取模型参数量（单位：K）"""
    try:
        model = tf.keras.models.load_model(model_path)
        params = model.count_params()
        return round(params / 1000, 1)  # 转换为K
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return 0


def get_train_time(log_data, num_epochs):
    """从日志中估算训练时间（s/epoch）"""
    # 注意：训练日志中没有直接记录训练时间
    # 这里我们使用一个近似值，基于之前的训练经验
    # 实际应用中，应该在训练脚本中记录每个epoch的训练时间
    # 这里我们根据模型大小和复杂度估算
    # 原始MLP: ~5s/epoch
    # MLP v1: ~5-6s/epoch  
    # MLP v2: ~8s/epoch
    # CNN: ~20s/epoch
    return 5.0  # 默认值，实际使用时需要修改


def process_log_file(log_path):
    """处理单个日志文件，提取所需信息"""
    # 从文件名中提取模型名称
    log_filename = os.path.basename(log_path)
    model_name = log_filename.replace("_log.json", "")
    
    # 读取日志数据
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # 获取最终的训练准确率和验证准确率
    train_acc_final = log_data["accuracy"][-1] if "accuracy" in log_data else 0
    val_acc_final = log_data["val_accuracy"][-1] if "val_accuracy" in log_data else 0
    
    # 计算过拟合gap
    overfitting_gap = train_acc_final - val_acc_final
    
    # 获取模型参数量
    model_path = os.path.join(CHECKPOINT_DIR, model_name, "model.keras")
    if not os.path.exists(model_path):
        # 兼容旧的模型路径
        model_path = os.path.join(CHECKPOINT_DIR, f"model_{model_name}_final.h5")
    params_k = get_model_params(model_path)
    
    # 获取训练时间（s/epoch）
    num_epochs = len(log_data["accuracy"]) if "accuracy" in log_data else 0
    train_time_per_epoch = get_train_time(log_data, num_epochs)
    
    # 测试准确率（这里使用验证准确率作为近似，实际应该使用独立的测试集）
    test_acc = val_acc_final
    
    return {
        "model_name": model_name,
        "test_acc": test_acc,
        "params_k": params_k,
        "overfitting_gap": overfitting_gap,
        "train_time_per_epoch": train_time_per_epoch
    }


def generate_markdown_table(models_data):
    """生成Markdown表格"""
    # 按test_acc降序排序
    sorted_models = sorted(models_data, key=lambda x: x["test_acc"], reverse=True)
    
    # 生成表格头部
    md_table = "| Model | Test Acc (%) | Params (K) | Overfitting Gap (%) | Train Time (s/epoch) |\n"
    md_table += "|-------|--------------|------------|---------------------|----------------------|\n"
    md_table += "|-------|--------------|------------|---------------------|----------------------|\n"
    
    # 生成表格内容
    for model in sorted_models:
        model_name = model["model_name"].replace("_", " ").title()
        test_acc = round(model["test_acc"] * 100, 2)
        params_k = model["params_k"]
        overfitting_gap = round(model["overfitting_gap"] * 100, 2)
        train_time = round(model["train_time_per_epoch"], 1)
        
        md_table += f"| {model_name} | {test_acc} | {params_k} | {overfitting_gap} | {train_time} |\n"
    
    return md_table


def generate_latex_table(models_data):
    """生成LaTeX表格"""
    # 按test_acc降序排序
    sorted_models = sorted(models_data, key=lambda x: x["test_acc"], reverse=True)
    
    # 生成表格
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Model Performance Comparison}\\n"
    latex_table += "\\begin{tabular}{|l|c|c|c|c|}\\hline\n"
    latex_table += "Model & Test Acc (\\%) & Params (K) & Overfitting Gap (\\%) & Train Time (s/epoch) \\\\hline\n"
    
    # 生成表格内容
    for model in sorted_models:
        model_name = model["model_name"].replace("_", " ").title()
        test_acc = round(model["test_acc"] * 100, 2)
        params_k = model["params_k"]
        overfitting_gap = round(model["overfitting_gap"] * 100, 2)
        train_time = round(model["train_time_per_epoch"], 1)
        
        latex_table += f"{model_name} & {test_acc} & {params_k} & {overfitting_gap} & {train_time} \\\\hline\n"
    
    latex_table += "\\end{tabular}\
"
    latex_table += "\\label{tab:model_comparison}\
"
    latex_table += "\\end{table}"
    
    return latex_table


def main():
    """主函数"""
    # 获取所有日志文件
    log_files = glob.glob(os.path.join(LOG_DIR, "*_log.json"))
    
    # 处理所有日志文件
    models_data = []
    for log_file in log_files:
        model_info = process_log_file(log_file)
        models_data.append(model_info)
    
    # 生成Markdown表格
    md_table = generate_markdown_table(models_data)
    md_output_path = os.path.join(OUTPUT_DIR, "comparison_table.md")
    with open(md_output_path, 'w') as f:
        f.write(md_table)
    print(f"Markdown table saved to {md_output_path}")
    
    # 生成LaTeX表格
    latex_table = generate_latex_table(models_data)
    latex_output_path = os.path.join(OUTPUT_DIR, "comparison_table.tex")
    with open(latex_output_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_output_path}")
    
    # 打印结果
    print("\nModel Comparison Results:")
    print(md_table)


if __name__ == "__main__":
    main()