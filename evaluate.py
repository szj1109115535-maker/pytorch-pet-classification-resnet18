import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

def plot_confusion_matrix(model, test_loader, device, save_img="/content/oxford_confusion_matrix.png"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="计算混淆矩阵"):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    max_val = np.diag(cm).max()

    plt.figure(figsize=(10,10), dpi=120)
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=max_val)
    plt.title("OxfordIIITPet Confusion Matrix", fontsize=10)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_img, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 混淆矩阵保存至 {save_img}")
