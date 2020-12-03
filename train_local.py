import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import MobileNetV21
from model_local import MobileNetV2
import torchvision.models as module
from tqdm import tqdm
from torchstat import stat
from matplotlib import pyplot as plt

nn.ConvTranspose2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


data_root = os.path.abspath(os.path.join(os.getcwd(), "/media/axxddzh/database"))  # get data root path
image_path = data_root + "/imagenet/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path+"train/ILSVRC2012_img_train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)


batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8,pin_memory = True)

validate_dataset = datasets.ImageFolder(root=image_path + "val/ILSVRC2012_img_val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8,pin_memory = True)

net = MobileNetV21()
stat(net, (3, 224, 224))
# net = module.mobilenet_v2(pretrained=False)
# load pretrain weights
# model_weight_path = "./mobilenet_v2-b0353104 (1).pth"
# pre_weights = torch.load(model_weight_path)
# # # delete classifier weights
# # pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
# net.load_state_dict(pre_weights, strict=False)
#
# # freeze features weights
# for param in net.features.parameters():
#     param.requires_grad = False

net.to(device)
lr_list = []
LR = 0.001
optimizer = optim.Adam(net.parameters(),lr = LR)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)

loss_function = nn.CrossEntropyLoss()

best_acc = 0.0
save_path = '.\\MobileNetV2.pth'
net.load_state_dict(torch.load(save_path), strict=False)
for epoch in range(150):
    # train

    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    net.train()
    running_loss = 0.0
    for step, data in enumerate(tqdm(train_loader), start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        print("train loss: {:.4f}".format(loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        plt.plot(range(len(lr_list)), lr_list, color='r')
    scheduler.step()
print('Finished Training')
# net.eval()
# acc = 0.0
# with torch.no_grad():
#         for val_data in tqdm(validate_loader):
#             val_images, val_labels = val_data
#             outputs = net(val_images.to(device))  # eval model only have last output layer
#             # loss = loss_function(outputs, test_labels)
#             predict_y = torch.max(outputs, dim=1)[1]
#             acc += (predict_y == val_labels.to(device)).sum().item()
#         val_accurate = acc / val_num
#         print('test_accuracy: %.3f' %(val_accurate))
