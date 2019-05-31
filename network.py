import torch
import torch.nn as nn
import torch.nn.functional as F

class H_swish(nn.Module):
    def __init__(self):
        super(H_swish, self).__init__()

    def forward(self, x):
        x = x * F.relu6(x + 3, inplace=True) / 6
        return x

# from https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
class SEModule(nn.Module):
    def __init__(self,in_channel,reduction=4):
        super(SEModule, self).__init__()
        #self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.linear1=nn.Linear(in_channel,in_channel//reduction,bias=True)
        self.no_linear1=nn.ReLU(inplace=True)
        self.linear2=nn.Linear(in_channel//reduction,in_channel,bias=True)
        self.no_linear2=nn.Sigmoid()

    def forward(self, x):
        y=F.avg_pool2d(x,kernel_size=x.size()[2:4])
        y=y.permute(0,2,3,1)
        y=self.linear1(y)
        y=self.no_linear1(y)
        y=self.linear2(y)
        y=self.no_linear2(y)
        y=y.permute(0,3,1,2)
        y=x*y

        return y

class extra_layer(nn.Module):
    def __init__(self,kernel_size,in_size,out_size,expand_size,no_linear,semodule=None,stride=1):
        super(extra_layer, self).__init__()
        self.semodule=False if semodule is None else True

        #self.no_linear = no_linear
        # if no_linear=="HS":
        #     self.no_linear=H_swish()
        # if no_linear=="RE":
        #     self.no_linear=nn.ReLU(inplace=True)

        self.connect=True if stride==1 and in_size==out_size else False

        if self.semodule:
            self.conv_withSe=nn.Sequential(
                nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False), # pointWise convolution(逐像素卷积)
                nn.BatchNorm2d(expand_size),
                no_linear,
                nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, padding=kernel_size // 2,
                          stride=stride,groups=expand_size, bias=False) , # depthWise convolution(逐层卷积)
                nn.BatchNorm2d(expand_size),
                SEModule(expand_size),
                no_linear,
                nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False) , # pointWise-linear
                nn.BatchNorm2d(out_size)
            )
        #self.features.append(self.conv_withSe)
        else:
            self.conv_withoutSe=nn.Sequential(
                nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False), # pointWise convolution(逐像素卷积)
                nn.BatchNorm2d(expand_size),
                no_linear,
                nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, padding=kernel_size // 2,
                          stride=stride,groups=expand_size, bias=False) , # depthWise convolution(逐层卷积)
                nn.BatchNorm2d(expand_size),
                no_linear,
                nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False) , # pointWise-linear
                nn.BatchNorm2d(out_size)
            )
            #self.features.append(self.conv_withoutSe)

    def forward(self, x):
        if self.semodule:
            out=self.conv_withSe(x)
        else:
            out=self.conv_withoutSe(x)

        if self.connect:
            out=x+out

        return out

class MobileNetV3_Large(nn.Module):
    def __init__(self,num_classes=10):
        super(MobileNetV3_Large, self).__init__()

        self.num_classes=num_classes
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.h_swishs1=H_swish()
        self.bnect=nn.Sequential(
            extra_layer(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            extra_layer(3, 16, 24, 64, nn.ReLU(inplace=True), None, 2),
            extra_layer(3, 24, 24, 72, nn.ReLU(inplace=True), None, 1),
            extra_layer(5, 24, 40, 72, nn.ReLU(inplace=True), "SE", 2),
            extra_layer(5, 40, 40, 120, nn.ReLU(inplace=True), "SE", 1),
            extra_layer(5, 40, 40, 120, nn.ReLU(inplace=True), "SE", 1),
            extra_layer(3, 40, 80, 240, H_swish(),None, 2),
            extra_layer(3, 80, 80, 200, H_swish(), None, 1),
            extra_layer(3, 80, 80, 184, H_swish(), None, 1),
            extra_layer(3, 80, 80, 184, H_swish(), None, 1),
            extra_layer(3, 80, 112, 480, H_swish(), "SE", 1),
            extra_layer(3, 112, 112, 672, H_swish(), "SE", 1),
            extra_layer(5, 112, 160, 672, H_swish(), "SE", 1),
            extra_layer(5, 160, 160, 672, H_swish(), "SE", 2),
            extra_layer(5, 160, 160, 960, H_swish(), "SE", 1),
            )
        self.conv2=nn.Conv2d(160,960,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=nn.BatchNorm2d(960)
        self.h_swishs2=H_swish()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.bn3=nn.BatchNorm2d(960)
        self.h_swishs3 = H_swish()
        self.conv3=nn.Linear(960,1280)
        self.h_swishs4=H_swish()
        self.linear=nn.Linear(1280,self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.h_swishs1(x)
        x = self.bnect(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.h_swishs2(x)
        x = self.avg_pool(x)
        x = self.bn3(x)
        x = self.h_swishs3(x)
        x=x.view(x.size(0),-1)
        x = self.conv3(x)
        x = self.h_swishs4(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
   net=MobileNetV3_Large()
   #print(net)
   x=torch.randn(2,3,224,224)
   y=net(x)
   print(y.size())

