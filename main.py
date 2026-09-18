import torch
from dataset import get_dataloaders
from model import build_resnet18
from train import train_model
from evaluate import plot_confusion_matrix
from gradcam_viz import plot_gradcam_figure

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    train_loader, test_loader, test_ds = get_dataloaders()
    model = build_resnet18(num_classes=37).to(device)

    # 训练
    model = train_model(model, train_loader, test_loader, device, epochs=11)
    model.eval()

    # 混淆矩阵
    plot_confusion_matrix(model, test_loader, device)

    # GradCAM可视化
    plot_gradcam_figure(model, test_ds, device)

if __name__ == "__main__":
    main()
