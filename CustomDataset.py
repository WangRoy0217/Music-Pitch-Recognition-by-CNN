from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

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
        img.show()

        if self.transform is not None:
            img = self.transform(img) #是否進行transform
        return img,label  #return很關鍵，return回哪些內容，那麼我們在訓練時迴圈讀取每個batch時，就能獲得哪些內容

    def __len__(self): #這個函式也必須要寫，它返回的是資料集的長度，也就是多少張圖片，要和loader的長度作區分
        return len(self.imgs)

#根據自己定義的那個勒MyDataset來建立資料集！注意是資料集！而不是loader迭代器
train_data=MyDataset(dir='./truth.txt',rootdir='./music_train/', transform=transforms.ToTensor())
train_loader=DataLoader(dataset=train_data,batch_size=50,shuffle=False)
for imgs, lbls in train_loader:
    print ('label %s'%lbls.data)  # batch_size*3*224*224

    break
