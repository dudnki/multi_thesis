import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
def main():
    # =========================
    # 1. 설정
    # =========================
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-3
    IMG_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # 2. 데이터셋 & 전처리
    # =========================
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = OxfordIIITPet(root="./data", split="trainval", transform=train_transform, download=True)
    test_dataset = OxfordIIITPet(root="./data", split="test", transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    NUM_CLASSES = len(train_dataset.classes)

    # =========================
    # 3. 모델 정의
    # =========================
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # =========================
    # 4. 학습 및 평가 함수
    # =========================
    def train_one_epoch():
        model.train()
        running_loss = 0
        running_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_correct / len(train_loader.dataset) * 100
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate():
        model.eval()
        running_loss = 0
        running_correct = 0
        
        
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = running_correct / len(test_loader.dataset) * 100
        return epoch_loss, epoch_acc

    # =========================
    # 5. 학습 루프
    # =========================

    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = evaluate()

        print(f"[Epoch {epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "classification/best_pet_classifier.pth")

    print(f"최고 검증 정확도: {best_acc:.2f}%")



if __name__ == "__main__":
    main()