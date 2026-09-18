# OxfordIIITPet ResNet18 宠物分类项目

基于PyTorch + ResNet18迁移学习实现37类宠物品种分类，包含模型训练、混淆矩阵评估、Grad-CAM可解释性可视化全流程。

## 🚀 一键复现
```bash
# 1. 安装依赖
pip install torch torchvision tqdm scikit-learn grad-cam matplotlib
# 2. 一键运行全流程
python main.py

## 📊 实验结果
- 数据集：OxfordIIITPet 37类猫狗品种
- 模型：ResNet18 迁移学习（ImageNet预训练骨干）
- 训练配置：Adam优化器，lr=1e-4，batch_size=32，11轮epoch
- 测试集最佳准确率：**89.07%**

### 训练曲线
![训练曲线](train_curve.png)

### 混淆矩阵
![混淆矩阵](oxford_confusion_matrix.png)

### Grad-CAM可解释性可视化
![GradCAM](oxford_gradcam_correct_wrong.png)

