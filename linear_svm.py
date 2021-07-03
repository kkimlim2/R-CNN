## linear svm의 경우 하나의 클래스만 구분 가능
## 만약 softmax 함수를 쓴다면 여러 개의 클래스를 동시에 구분 가능
## 각 클래스 별로 svm을 생성하는 것보다 softmax를 이용하는 것이 효율적
## 어렵게 생각하지 말고 그냥 softmax로 구현

import torch
import torch.nn as nn
import torchvision.models as models

from selectiveSearch import region_proposal
from customdata import CustomDataset, CustomSampler

from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.alexnet()
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 4)
model = model.to(device)

model.load_state_dict(torch.load('data/content/BCCD/fine_tuned.pth'))

for e, param in enumerate(model.parameters()):
    if e == 15: break
    param.requires_grad = False

model.classifier[6] = nn.Linear(in_features=4096, out_features=4, bias=True)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

print("모델 준비 됨")
## positive, negative에 대해서 다시 정의해야함
train_images, train_labels = region_proposal('classify')
print("input 준비 됨")

positive_num = 8
negative_num = 8
batch_size = positive_num + negative_num

dataset = CustomDataset(train_images, train_labels)
sampler = CustomSampler(train_labels, positive_num, negative_num)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

total_batch = sampler.iter

epochs = 15
print("훈련 시작")
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(dataloader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        model.train()
        optimizer.zero_grad()
        out = model(imgs.permute(0, 3, 1,
                                 2))  ## pytorch의 경우 인자의 순서가 (batchsize, channel, height, width) 우리는 보통 (batchsize,height,width,channel)이라고 생각하는데
        labels = labels.squeeze()  ## crossentropy input size 가 (N,C) 배치사이즈, 클래스 수 target size는 (N) 이 돼야해서 squeeze로 차원 변경
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print(f'[Epoch:{epoch + 1}] cost = {avg_cost}')
print('Learning Finished!')

torch.save(model.state_dict(), 'data/content/BCCD/linear_svm.pth') ## 가중치들 저장