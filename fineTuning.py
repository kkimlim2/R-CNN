import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

from selectiveSearch import region_proposal
from customdata import CustomSampler,CustomDataset

positive_num = 32
negative_num = 96
batch_size = positive_num + negative_num

train_images, train_labels = region_proposal('finetune')
print('region_proposal ended')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.alexnet(pretrained=True)
num_features = model.classifier[6].in_features

model.classifier[6] = nn.Linear(num_features, 4)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

dataset = CustomDataset(train_images, train_labels)
sampler = CustomSampler(train_labels, positive_num, negative_num)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

total_batch = sampler.iter

epochs = 15
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(dataloader):
        imgs, labels = data

        imgs = imgs.to(device)
        labels = labels.to(device)

        model.train()
        optimizer.zero_grad()

        print(imgs.shape)
        out = model(imgs.permute(0, 3, 1,2))  ## pytorch의 경우 인자의 순서가 (batchsize, channel, height, width) 우리는 보통 (batchsize,height,width,channel)이라고 생각하는데
        labels = labels.squeeze()  ## crossentropy input size 가 (N,C) 배치사이즈, 클래스 수 target size는 (N) 이 돼야해서 squeeze로 차원 변경
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print(f'[Epoch:{epoch + 1}] cost = {avg_cost}')
print('Learning Finished!')

torch.save(model.state_dict(), 'data/content/BCCD/fine_tuned.pth')  ## 가중치들 저장
