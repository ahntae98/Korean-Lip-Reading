#모듈 임포트
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms , models , datasets
from PIL import Image
import os
import glob
from tqdm import tqdm  # tqdm 임포트
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.font_manager as fm
import random
from torch.utils.data import random_split
from sklearn.model_selection import KFold

# 나눔글꼴 경로 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 폰트 이름 가져오기
font_name = fm.FontProperties(fname=font_path).get_name()

# 폰트 설정
plt.rc('font', family=font_name)

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터셋 클래스 정의
class LipReadingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.sequences = []
        self.labels = []
        self.sequence_length = 10  # 시퀀스 길이 고정
        self._load_data()

    def _load_data(self):
            for word in os.listdir(self.data_dir):
                word_path = os.path.join(self.data_dir, word)
                if os.path.isdir(word_path):
                    for speaker in os.listdir(word_path):
                        speaker_path = os.path.join(word_path, speaker)
                        if os.path.isdir(speaker_path):
                            for seq in os.listdir(speaker_path):
                                seq_path = os.path.join(speaker_path, seq)
                                if os.path.isdir(seq_path):
                                    sequence_images = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
                                    if len(sequence_images) >= 5:  # 최소 5장의 이미지가 있는 시퀀스만 사용
                                        self.sequences.append(sequence_images)
                                        self.labels.append(word)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_paths = self.sequences[idx]
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        
        # 시퀀스 길이를 고정
        if len(images) < self.sequence_length:
            images += [images[-1]] * (self.sequence_length - len(images))  # 마지막 이미지를 반복하여 채우기
        elif len(images) > self.sequence_length:
            images = images[:self.sequence_length]  # 처음 10개의 이미지만 사용
        
        if self.transform:
            images = [self.transform(image) for image in images]
        
        # 시퀀스 이미지를 텐서로 변환하고 하나의 텐서로 결합
        images_tensor = torch.stack(images)
        
        label = self.labels[idx]
        return images_tensor, label
    
    # 데이터 전처리
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
# 학습 데이터 변환 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

# 검증 데이터 변환 정의
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


data_dir = 'trainset_new'  # 데이터 경로
full_dataset = LipReadingDataset(data_dir)

train_size = int(0.8*len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset,val_dataset = random_split(full_dataset,[train_size,val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transfrom = val_transform

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset,batch_size=4,shuffle=False,num_workers=2)

# 레이블을 정수로 변환하는 매핑 생성
label_to_idx = {label: idx for idx, label in enumerate(set(full_dataset.labels))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

def visualize_sample(sample, label):
    images = sample.permute(0, 1, 2, 3)  # (T, C, H, W) -> (T, C, H, W)

    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    fig.suptitle(f'Label: {label}')
    for idx, img in enumerate(images):
        img = img.permute(1, 2, 0)  # 정규화 해제 (C, H, W) -> (H, W, C)
        img = img * 0.5 + 0.5  # 정규화 해제
        axes[idx].imshow(img.numpy())
        axes[idx].axis('off')
    plt.show()

sample, label = full_dataset[0]
visualize_sample(sample, label)

import torch
import torch.nn as nn
import torchvision.models as models

# MobileNetV2 + LSTM + Dropout 모델 정의
class MobileNetV2LSTM(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.6):
        super(MobileNetV2LSTM, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        mobilenet.features = nn.Sequential(
            *list(mobilenet.features),
            nn.AdaptiveAvgPool2d((1, 1))  # 추가: AdaptiveAvgPool2d로 마지막 출력 크기를 (1, 1)로 만듦
        )
        self.mobilenet = mobilenet.features
        self.lstm = nn.LSTM(1280, 256, batch_first=True)  # MobileNetV2의 출력 크기 1280
        self.dropout = nn.Dropout(dropout_rate)  # Dropout 레이어 추가
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.mobilenet(x).squeeze(-1).squeeze(-1)  # (batch_size * seq_length, 1280, 1, 1) -> (batch_size * seq_length, 1280)
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = self.dropout(x)  # Dropout 적용
        x = self.fc(x[:, -1, :])
        return x
# # MobileNet + LSTM 모델 정의
# class MobileNetV3LSTM(nn.Module):
#     def __init__(self, num_classes):
#         super(MobileNetV3LSTM, self).__init__()
#         mobilenet = models.mobilenet_v3_small(pretrained=True)
#         mobilenet.features = nn.Sequential(
#             *list(mobilenet.features),
#             nn.AdaptiveAvgPool2d((1, 1))  # 추가: AdaptiveAvgPool2d로 마지막 출력 크기를 (1, 1)로 만듦
#         )
#         self.mobilenet = mobilenet.features
#         self.lstm = nn.LSTM(576, 256, batch_first=True, dropout=0.5)  # MobileNetV3 Small의 출력 크기 576, dropout 추가
#         self.dropout = nn.Dropout(0.5)  # dropout 추가
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.size()
#         x = x.view(batch_size * seq_length, c, h, w)
#         x = self.mobilenet(x).squeeze(-1).squeeze(-1)  # (batch_size * seq_length, 576, 1, 1) -> (batch_size * seq_length, 576)
#         x = x.view(batch_size, seq_length, -1)
#         x, _ = self.lstm(x)
#         x = self.dropout(x[:, -1, :])  # dropout 추가
#         x = self.fc(x)
#         return x


    # 모델 초기화
num_classes = len(label_to_idx)
model = MobileNetV2LSTM(num_classes).to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay= 1e-4)

# 학습 손실 기록
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 모델 학습
num_epochs = 100
patience = 5

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0
    model.train()
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, labels = images.to(device), torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation step
    model.eval()
    val_epoch_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), torch.tensor([label_to_idx[label] for label in val_labels], dtype=torch.long).to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_epoch_loss += val_loss.item()
            
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
    
    avg_val_loss = val_epoch_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping")
            break

print("모델 학습 완료")

# 모델 저장
torch.save(model.state_dict(), 'lip_reading_model_1.pth')

test_val_dir = 'testset_new'

test_val_set = LipReadingDataset(test_val_dir, transform=val_transform)
test_val_loader = DataLoader(test_val_set, batch_size=4, shuffle=False, num_workers=4)

# 평가 함수 정의
def evaluate_model(model, dataloader, label_to_idx, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    unknown_label_idx = num_classes  # Unknown label index
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = [label_to_idx.get(label, unknown_label_idx) for label in labels]  # Unknown label handling
            valid_indices = [i for i, label in enumerate(labels) if label != unknown_label_idx]
            
            if valid_indices:  # 유효한 인덱스가 있을 때만 평가
                images = images[valid_indices]
                labels = torch.tensor([label for label in labels if label != unknown_label_idx], dtype=torch.long).to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    if all_labels and all_preds:
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    else:
        accuracy = 0.0
        cm = np.zeros((num_classes, num_classes), dtype=int)
    
    return accuracy, cm, all_preds, all_labels

# 혼동 행렬을 20개의 라벨씩 나누는 함수 정의
def plot_confusion_matrix_in_chunks(cm, idx_to_label, chunk_size=20):
    labels = list(idx_to_label.values())
    num_labels = len(labels)
    
    for i in range(0, num_labels, chunk_size):
        chunk_labels = labels[i:i+chunk_size]
        chunk_cm = cm[i:i+chunk_size, i:i+chunk_size]
        
        if chunk_cm.size > 0:  # chunk_cm이 비어있지 않을 때만 시각화
            plt.figure(figsize=(10, 8))
            sns.heatmap(chunk_cm, annot=True, fmt='d', cmap='Blues', xticklabels=chunk_labels, yticklabels=chunk_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Labels {i+1} to {i+len(chunk_labels)})')
            plt.show()

# 모델 평가
num_classes = len(label_to_idx)
accuracy, cm, all_preds, all_labels = evaluate_model(model, test_val_loader, label_to_idx, num_classes)
print(f'Accuracy: {accuracy:.4f}')

# 혼동 행렬 시각화
plot_confusion_matrix_in_chunks(cm, idx_to_label)

# 각 클래스별 정확도 계산 및 출력
with np.errstate(divide='ignore', invalid='ignore'):
    class_accuracy = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))

#for idx, acc in enumerate(class_accuracy):
 #   print(f'Accuracy for class {idx_to_label[idx]}: {acc:.4f}')

# 모델의 예측과 실제 라벨 출력
print("Predictions and Actual Labels:")
for i, (pred, actual) in enumerate(zip(all_preds, all_labels)):
    print(f"Sample {i+1}: Predicted: {idx_to_label[pred]}, Actual: {idx_to_label[actual]}")

# 모델의 출력 확인rate(zip(all_preds, all_labels)):
    print(f"Sample {i+1}: Predicted: {idx_to_label[pred]}, Actual: {idx_to_label[actual]}")

# 모델의 출력 확인
with torch.no_grad():
    output = model(sample.unsqueeze(0).to(device))
    _, pred = torch.max(output, 1)
    pred_label = idx_to_label[pred.item()]
    actual_label = label

print(f"Predicted label: {pred_label}, Actual label: {actual_label}")

# 혼동 행렬 및 예측 결과 저장
np.save('confusion_matrix.npy', cm)
np.save('all_preds.npy', np.array(all_preds))
np.save('all_labels.npy', np.array(all_labels))