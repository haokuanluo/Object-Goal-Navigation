import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import UNet
import habitat
import habitat_sim
import habitat_sim.utils
import habitat_sim.utils.data
from habitat_sim.utils.data import ImageExtractor
from torchvision import utils
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from os import path
from dataset import SemanticSegmentationDataset
import signal
from contextlib import contextmanager
import time
import os
from maskRCNN import get_model_instance_segmentation, get_pred, get_originalmodel
from maskdataset import MaskRCNNDataset,MaskRCNNDatasetPresaved
from engine import train_one_epoch, evaluate
from coco_utils import collate_fn





#training: trainScenes, Batchsize, numepochs,savcheck


model_path = "semantic/models/model_best.rcnn"
#model = get_model_instance_segmentation(22) # TODO: modifi!!!!!!!!!!!!
model = get_originalmodel()
orimodel = True



# Replace with the path to your scene file
dataPath = []
for i in range(16):
    dataPath.append("semantic/dataset/scenedata{}.p".format(i))
numData = 14

#trainScenes = 1
sav_check = 20
scene_switch = 10
BATCH_SIZE = 4
repetition = 100
scene_st = 0
loadModel = True    #TODO: very important!
save_path = "semantic/images/eval16/"    # TODO: very important!!!
mode = "eval"               #TODO: very important !!!!!!!!!!!!!
doEval = False
num_epochs = scene_switch * numData * repetition + 1
if mode == "eval":
    BATCH_SIZE = 2
    num_epochs = 0
    scene_st = numData
    loadModel = True
    doEval = True
if mode == "overfit":
    repetition = 10
    sav_check = 1000000
    loadModel = False
    doEval = True
    num_epochs = scene_switch * numData * repetition + 1
if mode == 'batchtest':
    trainScenes = 0
    scene_st = 0
    num_epochs = 1000
    BATCH_SIZE = 4
    sav_check = 1000000
    loadModel = True
    doEval = False

print(num_epochs)



if not path.exists(save_path):
    os.makedirs(save_path)

if path.exists(model_path) and loadModel and not orimodel:
    print("model loaded")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

def get_dataloader(dataPath):


    dataset = MaskRCNNDatasetPresaved(dataPath)

    # Create a Dataloader to batch and shuffle our data
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_fn)



val_check = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, # 0.005
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)




dataloader = get_dataloader(dataPath[scene_st])
for epoch in range(num_epochs):
    print_freq = 10000
    if epoch % scene_switch == 0 or epoch % scene_switch == scene_switch-1:
        print_freq = 200
    train_one_epoch(model, optimizer, dataloader, device, epoch,
                    print_freq=print_freq)
    # Evaluate the model on validation set
    #if epoch % val_check == 1:
    #    print(f"iter: {epoch}, train loss: {epoch_loss}")
    if epoch % sav_check == 0 and loadModel:
        print("model saved")
        torch.save(model.state_dict(), model_path)
    if epoch % scene_switch == scene_switch-1:
        scene_st += 1
        if scene_st >= numData:
            scene_st = 0
        dataloader = get_dataloader(dataPath[scene_st])

picid = 0
def show_batch(sample_batch):
    def show_row(imgs, batch_size, img_type):
        global picid
        plt.figure(figsize=(12, 8))
        for i, img in enumerate(imgs):
            ax = plt.subplot(1, batch_size, i + 1)
            ax.axis("off")
            if img_type == 'rgb':
                plt.imshow(img.numpy().transpose(1, 2, 0))
            elif img_type in ['truth', 'target', 'prediction']:
                plt.imshow(img.numpy())
        fn = save_path+ 'train_val_{}.png'.format(
            picid)
        picid = picid + 1
        plt.savefig(fn)
        plt.close()

    batch_size = len(sample_batch['rgb'])
    for k in sample_batch.keys():
        show_row(sample_batch[k], batch_size, k)


with torch.no_grad():
    #model.to('cpu')
    model.eval()
    _, batch = next(enumerate(dataloader))
    mask_pred = get_pred(model,batch)
    #imgs = {}
    #imgs['rgb'] = batch[0]
    #mask_pred = model(batch['rgb'])
    #mask_pred = F.softmax(mask_pred, dim=1)
    #mask_pred = torch.argmax(mask_pred, dim=1)

    #batch['prediction'] = mask_pred
    if doEval:
        show_batch(mask_pred)
