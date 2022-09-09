import torch.utils.data
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision                                       
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class MyDataset(torch.utils.data.Dataset): #建立自己的類：MyDataset,這個類是繼承的torch.utils.data.Dataset
    def __init__(self,dir, rootdir,transform=None, target_transform=None): #初始化一些需要傳入的引數
        fh = open(dir, 'r') #按照傳入的路徑和txt文字引數，開啟這個文字，並讀取內容
        imgs = []                      #建立一個名為img的空列表，一會兒用來裝東西
        for line in fh:                #按行迴圈txt文字中的內容
            line = line.rstrip()       # 刪除 本行string 字串末尾的指定字元，這個方法的詳細介紹自己查詢python
            words = line.split()   #通過指定分隔符對字串進行切片，預設為所有的空字元，包括空格、換行、製表符等
            imgs.append((rootdir+words[0],int(words[1]))) #把txt裡的內容讀入imgs列表儲存，具體是words幾要看txt內容而定
                                        # 很顯然，根據我剛才截圖所示txt的內容，words[0]是圖片資訊，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):    #這個方法是必須要有的，用於按照索引讀取每個元素的具體內容
        fn, label = self.imgs[index] #fn是圖片path #fn和label分別獲得imgs[index]也即是剛才每行中word[0]和word[1]的資訊
        img = Image.open(fn).convert('RGB') #按照path讀入圖片from PIL import Image # 按照路徑讀取圖片

        if self.transform is not None:
            img = self.transform(img) #是否進行transform
        return img,label  #return很關鍵，return回哪些內容，那麼我們在訓練時迴圈讀取每個batch時，就能獲得哪些內容

    def __len__(self): #這個函式也必須要寫，它返回的是資料集的長度，也就是多少張圖片，要和loader的長度作區分
        return len(self.imgs)




train_dataset = MyDataset(dir='./truth.txt',rootdir='./music_train/', transform=transforms.ToTensor())

test_dataset = MyDataset(dir='./test.txt',rootdir='./music_test/', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=10, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10, 
                                          shuffle=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
#nn.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
resnet = ResNet(ResidualBlock, [3, 3, 3])
resnet.cuda()

criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    
for epoch in range(80):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, 80, i+1, 500, loss.data[0]))

    if (epoch+1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
 
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.cuda())
    outputs = resnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)

correct += (predicted.cpu() == labels).sum()

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

torch.save(resnet.state_dict(), 'resnet.pkl')
