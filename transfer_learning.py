import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

import matplotlib.pyplot as plt # 그래프를 그리기 위해 상단에 추가

# --- [설정 섹션] ---
# 학습 모드 선택: 1 또는 2
# 1: Fine-tuning (전체 레이어 학습)
# 2: Fixed Feature Extractor (마지막 레이어만 학습)
SCENARIO = 1 

# 데이터셋 경로 (개미/벌 데이터셋이 있는 경로로 수정하세요)
data_dir = '/home/songah/resnet/transfer_learning/hymenoptera_data'
# ------------------

# 1. 데이터 전처리 및 로드
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 랜덤 크롭
        transforms.RandomHorizontalFlip(), # 랜덤 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 정규화 기준
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'val']}

### CIFAR10 dataset
image_datasets = {
    'train': datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train']),
    'val': datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# class_names = image_datasets['train'].classes
### CIFAR10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. 학습용 공통 함수 정의
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # 가장 성능이 좋은 가중치를 저장하기 위한 변수
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # --- 추가: 기록을 위한 리스트 생성 ---
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터 로드
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 파라미터 경사도 초기화
                optimizer.zero_grad()

                # 순전파(Forward)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계에서만 역전파 및 최적화 수행
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # --- 추가: 각 에폭의 결과 기록 ---
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            # -------------------------------

            # 검증 단계에서 정확도가 가장 높으면 가중치 복사
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 가장 성능 좋았던 가중치를 모델에 입혀서 반환
    model.load_state_dict(best_model_wts)
    return model, history

# 3. 모델 불러오기 및 설정
# 사전 학습된 ResNet18 불러오기
model_resnet = models.resnet18(weights='IMAGENET1K_V1')

if SCENARIO == 1:
    print("선택된 시나리오: 1 (Fine-tuning 전체 미세 조정)")
    # 모든 가중치를 학습 가능하도록 둠 (기본값)
    # 마지막 레이어만 내 데이터의 클래스 수(여기서는 2)에 맞게 변경
    num_ftrs = model_resnet.fc.in_features
    
    # model_resnet.fc = nn.Linear(num_ftrs, len(class_names))
    ### CIFAR10 dataset
    model_resnet.fc = nn.Linear(num_ftrs, 10)
    
    model_resnet = model_resnet.to(device)
    optimizer = optim.SGD(model_resnet.parameters(), lr=0.001, momentum=0.9)

elif SCENARIO == 2:
    print("선택된 시나리오: 2 (Fixed Feature Extractor 고정)")
    # 1단계: 모든 레이어의 가중치를 고정 (기울기 계산 중지)
    for param in model_resnet.parameters():
        param.requires_grad = False
    
    # 2단계: 마지막 레이어를 교체 (새로운 레이어는 자동으로 학습 가능 상태가 됨)
    num_ftrs = model_resnet.fc.in_features
    
    # model_resnet.fc = nn.Linear(num_ftrs, len(class_names))
    ### CIFAR10 dataset
    model_resnet.fc = nn.Linear(num_ftrs, 10)
    
    model_resnet = model_resnet.to(device)
    # 3단계: 최적화 도구에 '마지막 레이어의 파라미터'만 전달하여 이것만 학습되게 함
    optimizer = optim.SGD(model_resnet.fc.parameters(), lr=0.001, momentum=0.9)

# 공통 설정: 손실 함수와 스케줄러
criterion = nn.CrossEntropyLoss()
# 7 에폭마다 학습률에 0.1을 곱해줌
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 4. 학습 시작
model_final, history = train_model(model_resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

# 5. output 폴더가 없으면 생성
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"폴더 생성 완료: {output_dir}")

# 6. 베스트 모델 가중치 저장 (output 폴더 내)
weights_path = os.path.join(output_dir, 'best_resnet18_weights_CIFAR10.pth')
torch.save(model_final.state_dict(), weights_path)
print(f"Best 가중치 저장 완료: {weights_path}")

# 7. Loss 및 Accuracy 그래프 그리기 및 저장
plt.figure(figsize=(12, 5))

# (1) Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='red')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True) # 눈금선 추가

# (2) Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='blue')
plt.plot(history['val_acc'], label='Val Acc', color='red')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 8. 이미지 파일 저장 (output 폴더 내)
plt.tight_layout()
graph_path = os.path.join(output_dir, 'learning_curve_CIFAR10.png')
plt.savefig(graph_path) 
print(f"학습 그래프 저장 완료: {graph_path}")