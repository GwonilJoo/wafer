# Wafer

2021 RTM 인턴십 과제

## 문제
반도체 wafer 이미지 불량 유형 분류

## 풀이
이미지 사이즈를 224x224로 변환 후, 두 개의 모델 비교 후 성능 측정
1. Custom CNN
2. Resnet18

## 실행 방법

### train
- default 실행
```
python train.py
```

- argument 지정 실행
```
python train.py \
--lr 1e-3 \
--num_epochs 10 \
--batch_size 128 \
--data_dir ./dataset/train \
--save_dir ./saved/resnet-pretrained_1e-3/
--random_seed 222
```

### test
- default 실행
```
python test.py
```

- argument 지정 실행
```
python test.py \
--batch_size 128 \
--data_dir ./dataset/test \
--weight_path ./saved/resnet-pretrained_1e-3/9_model.pt
```
