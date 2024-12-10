import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, init_weights=True):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 输入4张图像
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(64, 3, kernel_size=1)  

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c1 = F.relu(self.conv2(c1))
        p1 = self.pool1(c1)

        c2 = F.relu(self.conv3(p1))
        c2 = F.relu(self.conv4(c2))
        p2 = self.pool2(c2)

        c3 = F.relu(self.conv5(p2))
        c3 = F.relu(self.conv6(c3))
        p3 = self.pool3(c3)

        c4 = F.relu(self.conv7(p3))
        c4 = F.relu(self.conv8(c4))
        p4 = self.pool4(c4)

        c5 = F.relu(self.conv9(p4))
        c5 = F.relu(self.conv10(c5))

        u6 = self.upconv1(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = F.relu(self.conv11(u6))
        c6 = F.relu(self.conv12(c6))

        u7 = self.upconv2(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.relu(self.conv13(u7))
        c7 = F.relu(self.conv14(c7))

        u8 = self.upconv3(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.relu(self.conv15(u8))
        c8 = F.relu(self.conv16(c8))

        u9 = self.upconv4(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.relu(self.conv17(u9))
        c9 = F.relu(self.conv18(c9))

        output = self.conv19(c9)
        # 不使用激活函数，直接输出，以允许负值（法向量的分量可以为负）

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
