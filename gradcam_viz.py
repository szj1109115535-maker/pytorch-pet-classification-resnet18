import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from torchvision import transforms

def plot_gradcam_figure(model, test_ds, device, save_img="/content/oxford_gradcam_correct_wrong.png"):
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    correct_idx, wrong_idx = None, None

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_ori, label_gt = test_ds[idx]
            img_tensor = transform(img_ori).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred_cls = torch.argmax(output, dim=1).item()
            if pred_cls == label_gt and correct_idx is None:
                correct_idx = idx
            if pred_cls != label_gt and wrong_idx is None:
                wrong_idx = idx
            if correct_idx is not None and wrong_idx is not None:
                break

    def get_cam(img_ori, target_cls):
        img_tensor = transform(img_ori).unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(target_cls)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        img_np = np.array(img_ori.resize((224,224))).astype(np.float32)/255
        return show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.6)

    img_correct, label_correct = test_ds[correct_idx]
    img_wrong, label_wrong = test_ds[wrong_idx]

    pred_correct = torch.argmax(model(transform(img_correct).unsqueeze(0).to(device)), dim=1).item()
    pred_wrong = torch.argmax(model(transform(img_wrong).unsqueeze(0).to(device)), dim=1).item()

    cam_correct = get_cam(img_correct, pred_correct)
    cam_wrong = get_cam(img_wrong, pred_wrong)

    plt.figure(figsize=(10,10), dpi=120)
    plt.subplot(2,2,1)
    plt.imshow(img_correct.resize((224,224)))
    plt.title(f"Correct Sample | True:{label_correct}, Pred:{pred_correct}", fontsize=9)
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(img_wrong.resize((224,224)))
    plt.title(f"Wrong Sample | True:{label_wrong}, Pred:{pred_wrong}", fontsize=9)
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(cam_correct)
    plt.title("Grad-CAM (Correct)", fontsize=9)
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(cam_wrong)
    plt.title("Grad-CAM (Wrong)", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_img, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ GradCAM图保存至 {save_img}")
