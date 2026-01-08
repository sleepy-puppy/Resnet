import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. 모델 설정 (학습 때와 동일한 구조여야 합니다)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 클래스 이름 정의 (학습 데이터의 폴더 순서와 동일해야 함)
# class_names = ['ants', 'bees'] 
### CIFAR10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path):
    # ResNet18 구조 생성
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    
    # 학습 시 수정했던 대로 FC 레이어 출력 크기를 맞춤 (2개 클래스)
    # model.fc = nn.Linear(num_ftrs, len(class_names))
    ### CIFAR10 dataset
    model.fc = nn.Linear(num_ftrs, 10)
    
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # 추론 모드로 설정 (드롭아웃, 배치 정규화 등을 평가 모드로 전환)
    return model

# 2. 이미지 전처리 함수
def process_image(image_path):
    # 학습 시 val 데이터에 적용했던 것과 동일한 전처리 과정을 거쳐야 합니다.
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0) # 모델 입력을 위해 4차원(batch size 추가)으로 변환
    return image.to(device)

# 3. 추론 실행
def predict(model, image_path):
    inputs = process_image(image_path)
    
    with torch.no_grad(): # 추론 시에는 기울기 계산이 필요 없음
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        # Softmax를 통해 확률 확인 (선택 사항)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    result_idx = preds[0].item()
    result_class = class_names[result_idx]
    confidence = probs[0][result_idx].item() * 100

    print(f"이미지 경로: {image_path}")
    print(f"예측 결과: {result_class} ({confidence:.2f}%)")

if __name__ == "__main__":
    # 파일 경로 설정
    weights_path = './output/best_resnet18_weights_CIFAR10.pth'
    image_path = './test_CIFAR10.jpg'

    if os.path.exists(weights_path) and os.path.exists(image_path):
        trained_model = load_model(weights_path)
        predict(trained_model, image_path)
    else:
        if not os.path.exists(weights_path):
            print(f"에러: 가중치 파일({weights_path})이 없습니다. 학습을 먼저 진행하세요.")
        if not os.path.exists(image_path):
            print(f"에러: 테스트 이미지({image_path})가 폴더에 없습니다.")