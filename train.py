import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np

from model import resnet18_DW


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        # 计算测试样本的总数
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix ')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

def main():
    epochsArr =[]
    accArr = []
    lossArr =[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 32
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "tea_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet18_DW()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    # model_weight_path = "./resnet34-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # 预训练参数的位置

    # net_path = "./resnet34-pre.pth"
    # net_dict = net.state_dict()  # 网络层的参数
    # # # 需要加载的预训练参数
    # pretrained_dict = torch.load(net_path) # torch.load得到是字典，我们需要的是state_dict下的参数
    # # 删除pretrained_dict.items()中model所没有的东西
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict and k not in ['classifier.1.weight', 'classifier.1.bias']}
    # net_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
    # net.load_state_dict(net_dict)  # model加载dict中的数据，更新网络的初始值

    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    #in_channel后面为类别个数
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 50
    best_acc = 0.0
    save_path = './resnet18_DW.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        class_indict = {"0": "lower_tea", "1": "second_tea", "2": "senior_tea", "3": "super_tea"}
        label = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=4, labels=label)
        # 实例化混淆矩阵，这里NUM_CLASSES =
        # validate

        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            net.eval()
            val_bar = tqdm(validate_loader, file=sys.stdout)  # 进度条
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_labels = val_labels.to(device)
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                confusion.update(predict_y.cpu().numpy(), val_labels.cpu().numpy())

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        epochsArr.append(epoch + 1)
        accArr.append(val_accurate)
        lossArr.append(running_loss / train_steps)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('accArr：'+str(accArr))
    print('loss：'+str(lossArr))
    print('Finished Training')
    confusion.plot()
    confusion.summary()
    plt.plot(epochsArr, accArr, color='r', label='acc')  # r表示红色
    plt.plot(epochsArr, lossArr, color=(0, 0, 0), label='loss')  # 也可以用RGB值表示颜色
    plt.xlabel('epochs')  # x轴表示
    plt.ylabel('y label')  # y轴表示
    plt.title("chart")  # 图标标题表示
    plt.legend()  # 每条折线的label显示
    plt.savefig('test.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片


if __name__ == '__main__':
    main()
