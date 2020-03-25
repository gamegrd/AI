import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data as Data
import torchvision

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 1  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='../mnist/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='../mnist/',
    train=False  # 表明是测试集
)

# 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱数据，一般都打乱
)

# 进行测试
# 为节约时间，测试时只测试前2000个
#
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.test_labels[:2000]


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出
class Mish(nn.Module):
    """Mish激活函数"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(func.softplus(x)))


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1,
                 padding=0, dilation=1, bias=True):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数，即卷积核的个数
        :param kernel_size: 卷积核的尺寸,如果传整数则是正方形边长，tuple则是实际尺寸
        :param stride: 步长， default 1
        :param padding: 填充个数， default 0
        :param bias: 是否使用偏置， default True
        """
        super().__init__()

        # 镜像填充 + 卷积 + BatchNorm
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding=padding),

            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, bias=bias),

            nn.BatchNorm2d(out_channels, eps=0.001)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        return x


class MaxPool(nn.Module):

    def __init__(self, kernel_size, *, stride=1, padding=0):
        """
        :param kernel_size:
        :param stride:
        :param padding:
        镜面填充再进行平均池化
        """
        super().__init__()
        self.avg_pool = nn.Sequential(
            nn.ReflectionPad2d(padding=padding),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.avg_pool(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()

        # AdaptiveAvgPool2d 在每一层通道上进行自适应的平均池化
        # 传入的参数是要输出的每一层通道上二维数组的尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # AdaptiveMaxPool2d 在每一层通道上进行自适应的最大池化
        # 传入的参数是要输出的每一层通道上二维数组的尺寸
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = in_planes // ratio
        if mid_channels == 0:
            mid_channels = 1

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Tanh(),
            nn.Conv2d(mid_channels, in_planes, 1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.Tanh(),
        )

    def forward(self, x) -> torch.Tensor:
        # return_shape = input_shape

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        out = avg_out + max_out
        channel_attention = out.sigmoid()

        return x * channel_attention  # broadcasting


class SpacialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            BasicConv2d(in_channels=2, out_channels=1,
                        kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        # return_shape = input_shape

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)

        spacial_attention = self.conv(out)

        return x * spacial_attention  # broadcasting


class CBAM(nn.Module):
    """
    串联的注意力机制
    """

    def __init__(self, in_planes, *, ratio=16, kernel_size=3):
        super().__init__()

        self.channel_attention = ChannelAttention(in_planes=in_planes, ratio=ratio)
        self.spacial_attention = SpacialAttention(kernel_size=kernel_size)

        self.batch_norm = nn.BatchNorm2d(in_planes)

    def forward(self, x) -> torch.Tensor:
        # 返回经过注意力机制处理过的feature map，shape没有变化

        residual = x

        x = self.channel_attention(x)
        x = self.spacial_attention(x)

        x = x + residual

        x = self.batch_norm(x)
        x = x.relu()

        return x


class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        self.cbam1 = CBAM(in_planes=16)
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        self.cbam2 = CBAM(in_planes=32)
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.cbam1(x)
        x = self.conv2(x)  # 再通过conv2
        x = self.cbam2(x)
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练和测试
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        output = cnn(b_x)  # 先将数据放到cnn中计算output
        loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 应用梯度

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
