from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32, num_workers=2):
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_ds = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform
    )
    test_ds = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, test_ds
