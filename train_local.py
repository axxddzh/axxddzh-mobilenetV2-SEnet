import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from mobile_backbone import MobileBackbone
from model_local import MobileNetV2
import torchvision.models as module
from tqdm import tqdm
from torchstat import stat
from matplotlib import pyplot as plt
import pandas as pd
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

net = MobileBackbone()
# stat(net, (3, 300, 300))
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
clm=["train_loss","lr","val_acc"]
net.to(device)
lr_list = []
LR = 0.0001
optimizer = optim.Adam(net.parameters(),lr = LR)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma = 0.5)

loss_function = nn.CrossEntropyLoss()

best_acc = 0.0
if not os.path.exists("./save_weight"):
    os.makedirs("./save_weight")

start_epoch = 0
train_loss = []
val_map = []
resume = ""
if resume != "":
    checkpoint = torch.load(resume)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['train_loss']
    val_map = checkpoint['val_map']
    lr_list = checkpoint['lr_list']
    print("the training process from epoch{}...".format(start_epoch))


for epoch in range(150):
    # train
    if epoch < start_epoch:
        continue
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
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
        print("epoch:{},batch:{}/10010,train loss: {:.4f}".format(epoch+1,step,loss))
    print()
    loss_c = loss
    train_loss.append(loss_c.cpu().detach().numpy())
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
            save_files = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss':train_loss,
                "lr_list":lr_list,
                'val_map':val_map
            }
            torch.save(save_files, "./save_weight/ssd300-{}-{}.pth".format(epoch,val_accurate))
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        plt.plot(range(len(lr_list)), lr_list, color='r')
        val_map.append(val_accurate)
        df = pd.DataFrame([train_loss,lr_list,val_map],index=clm)
        df.to_csv('Result.csv')
        # plot loss and lr curve
        if len(train_loss) != 0 and len(lr_list) != 0:
            from plot_curve import plot_loss_and_lr

            plot_loss_and_lr(train_loss, lr_list)

        # plot mAP curve
        if len(val_map) != 0:
            from plot_curve import plot_map

            plot_map(val_map)
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
