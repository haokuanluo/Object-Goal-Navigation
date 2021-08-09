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
from maskRCNN import get_model_instance_segmentation, get_pred
from maskdataset import MaskRCNNDataset
from engine import train_one_epoch, evaluate
from coco_utils import collate_fn

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



#training: trainScenes, Batchsize, numepochs,savcheck
BATCH_SIZE = 2

model_path = "semantic/models/model_best.rcnn"
model = get_model_instance_segmentation(22)

#model = UNet(n_channels=3, n_classes=22)


# Replace with the path to your scene file
ALLSCENES = ['data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb', 'data/scene_datasets/mp3d/aayBHfsNo7d/aayBHfsNo7d.glb', 'data/scene_datasets/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb', 'data/scene_datasets/mp3d/pa4otMbVnkk/pa4otMbVnkk.glb', 'data/scene_datasets/mp3d/S9hNv5qa7GM/S9hNv5qa7GM.glb', 'data/scene_datasets/mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.glb', 'data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb', 'data/scene_datasets/mp3d/ac26ZMwG7aT/ac26ZMwG7aT.glb', 'data/scene_datasets/mp3d/gxdoqLR6rwA/gxdoqLR6rwA.glb', 'data/scene_datasets/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb', 'data/scene_datasets/mp3d/sKLMLpTHeUy/sKLMLpTHeUy.glb', 'data/scene_datasets/mp3d/vyrNrziPKCB/vyrNrziPKCB.glb', 'data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb', 'data/scene_datasets/mp3d/ARNzJeq3xxb/ARNzJeq3xxb.glb', 'data/scene_datasets/mp3d/gYvKGZ5eRqb/gYvKGZ5eRqb.glb', 'data/scene_datasets/mp3d/Pm6F8kyY3z2/Pm6F8kyY3z2.glb', 'data/scene_datasets/mp3d/SN83YJsR3w2/SN83YJsR3w2.glb', 'data/scene_datasets/mp3d/VzqfbhrpDEA/VzqfbhrpDEA.glb', 'data/scene_datasets/mp3d/29hnd4uzFmX/29hnd4uzFmX.glb', 'data/scene_datasets/mp3d/B6ByNegPMKs/B6ByNegPMKs.glb', 'data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb', 'data/scene_datasets/mp3d/pRbA3pwrgk9/pRbA3pwrgk9.glb', 'data/scene_datasets/mp3d/sT4fr6TAbpF/sT4fr6TAbpF.glb', 'data/scene_datasets/mp3d/wc2JMjhGNzB/wc2JMjhGNzB.glb', 'data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb', 'data/scene_datasets/mp3d/b8cTxDM8gDG/b8cTxDM8gDG.glb', 'data/scene_datasets/mp3d/HxpKQynjfin/HxpKQynjfin.glb', 'data/scene_datasets/mp3d/PuKPg4mmafe/PuKPg4mmafe.glb', 'data/scene_datasets/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb', 'data/scene_datasets/mp3d/WYY7iVyf5p8/WYY7iVyf5p8.glb', 'data/scene_datasets/mp3d/2n8kARJN3HM/2n8kARJN3HM.glb', 'data/scene_datasets/mp3d/cV4RVeZvu5T/cV4RVeZvu5T.glb', 'data/scene_datasets/mp3d/i5noydFURQK/i5noydFURQK.glb', 'data/scene_datasets/mp3d/PX4nDJXEHrG/PX4nDJXEHrG.glb', 'data/scene_datasets/mp3d/ULsKaCPVFJR/ULsKaCPVFJR.glb', 'data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb', 'data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.glb', 'data/scene_datasets/mp3d/D7G3Y4RVNrH/D7G3Y4RVNrH.glb', 'data/scene_datasets/mp3d/JeFG25nYj2p/JeFG25nYj2p.glb', 'data/scene_datasets/mp3d/q9vSo1VnCiC/q9vSo1VnCiC.glb', 'data/scene_datasets/mp3d/uNb9QFRL6hY/uNb9QFRL6hY.glb', 'data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb', 'data/scene_datasets/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb', 'data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.glb', 'data/scene_datasets/mp3d/JF19kD82Mey/JF19kD82Mey.glb', 'data/scene_datasets/mp3d/qoiz87JEwZ2/qoiz87JEwZ2.glb', 'data/scene_datasets/mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.glb', 'data/scene_datasets/mp3d/XcA2TqTSSAj/XcA2TqTSSAj.glb', 'data/scene_datasets/mp3d/5q7pvUzZiYa/5q7pvUzZiYa.glb', 'data/scene_datasets/mp3d/dhjEzFoUFzH/dhjEzFoUFzH.glb', 'data/scene_datasets/mp3d/jh4fc5c5qoQ/jh4fc5c5qoQ.glb', 'data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb', 'data/scene_datasets/mp3d/UwV83HsGsw3/UwV83HsGsw3.glb', 'data/scene_datasets/mp3d/YFuZgdQ5vWj/YFuZgdQ5vWj.glb', 'data/scene_datasets/mp3d/5ZKStnWn8Zo/5ZKStnWn8Zo.glb', 'data/scene_datasets/mp3d/E9uDoFAP3SH/E9uDoFAP3SH.glb', 'data/scene_datasets/mp3d/JmbYfDe2QKZ/JmbYfDe2QKZ.glb', 'data/scene_datasets/mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.glb', 'data/scene_datasets/mp3d/Uxmj2M2itWa/Uxmj2M2itWa.glb', 'data/scene_datasets/mp3d/YmJkqBEsHnH/YmJkqBEsHnH.glb', 'data/scene_datasets/mp3d/759xd9YjKW5/759xd9YjKW5.glb', 'data/scene_datasets/mp3d/e9zR4mvMWw7/e9zR4mvMWw7.glb', 'data/scene_datasets/mp3d/jtcxE69GiFV/jtcxE69GiFV.glb', 'data/scene_datasets/mp3d/r47D5H71a5s/r47D5H71a5s.glb', 'data/scene_datasets/mp3d/V2XKFyX4ASd/V2XKFyX4ASd.glb', 'data/scene_datasets/mp3d/yqstnuAEVhm/yqstnuAEVhm.glb', 'data/scene_datasets/mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb', 'data/scene_datasets/mp3d/EDJbREhghzL/EDJbREhghzL.glb', 'data/scene_datasets/mp3d/kEZ7cmS4wCh/kEZ7cmS4wCh.glb', 'data/scene_datasets/mp3d/rPc6DW4iMge/rPc6DW4iMge.glb', 'data/scene_datasets/mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.glb', 'data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.glb', 'data/scene_datasets/mp3d/8194nk5LbLH/8194nk5LbLH.glb', 'data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb', 'data/scene_datasets/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.glb', 'data/scene_datasets/mp3d/RPmz2sHmrrY/RPmz2sHmrrY.glb', 'data/scene_datasets/mp3d/VLzqgDo317F/VLzqgDo317F.glb', 'data/scene_datasets/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb', 'data/scene_datasets/mp3d/82sE5b5pLXE/82sE5b5pLXE.glb', 'data/scene_datasets/mp3d/fzynW3qQPVF/fzynW3qQPVF.glb', 'data/scene_datasets/mp3d/oLBMNvg9in8/oLBMNvg9in8.glb', 'data/scene_datasets/mp3d/rqfALeAoiTq/rqfALeAoiTq.glb', 'data/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.glb', 'data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.glb', 'data/scene_datasets/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb', 'data/scene_datasets/mp3d/GdvgFV5R1Z5/GdvgFV5R1Z5.glb', 'data/scene_datasets/mp3d/p5wJjkQkbXX/p5wJjkQkbXX.glb', 'data/scene_datasets/mp3d/s8pcmisQ38h/s8pcmisQ38h.glb', 'data/scene_datasets/mp3d/VVfe2KiqLaN/VVfe2KiqLaN.glb', 'data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb']
numScenes = len(ALLSCENES)
trainScenes = int(numScenes * 0.8)

#trainScenes = 1
sav_check = 20
scene_switch = 10

repetition = 100
scene_batch = 3 # TODO: could be modified
num_epochs = scene_switch * int(trainScenes / scene_batch) * repetition + 1
print(num_epochs)
scene_st = 0
loadModel = True
save_path = "semantic/images/eval8/"
mode = "train"
doEval = False
if mode == "eval":
    BATCH_SIZE = 4
    num_epochs = 0
    sav_check = 2
    scene_st = 75
    scene_batch = 1
if mode == "overfit":
    trainScenes = 0
    scene_st = 0
    num_epochs = 100
    BATCH_SIZE = 4
    print("batch size")
    sav_check = 1000000
    scene_batch = 1
    loadModel = False
if mode == 'batchtest':
    trainScenes = 0
    scene_st = 0
    num_epochs = 1000
    BATCH_SIZE = 4
    sav_check = 1000000
    scene_batch = 1
    loadModel = True
    doEval = False


if not path.exists(save_path):
    os.makedirs(save_path)

if path.exists(model_path) and loadModel:
    print("model loaded")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

def get_dataloader(SCENE_FILEPATH):
    global scene_st

    try:
        with time_limit(120):
            extractor = ImageExtractor(SCENE_FILEPATH, img_size=(256, 256),
                                       output=['rgba', 'semantic'],
                                       pose_extractor_name="panorama_extractor")
    except TimeoutException as e:
        print("Timed out!")
        exit(0)

    print(scene_st,trainScenes)
    while len(extractor) < 1:
        print("bad scene")
        extractor.close()
        scene_st += scene_batch
        if scene_st + scene_batch > trainScenes:
            scene_st = 0
        print(scene_st, trainScenes)
        try:
            with time_limit(120):
                extractor = ImageExtractor(ALLSCENES[scene_st:scene_st+scene_batch], img_size=(256, 256),
                                           output=['rgba', 'semantic'],
                                           pose_extractor_name="panorama_extractor")
        except TimeoutException as e:
            print("Timed out!")
            exit(0)

    dataset = MaskRCNNDataset(extractor,transforms=transforms.Compose([transforms.ToTensor()]))

    # Create a Dataloader to batch and shuffle our data
    return extractor,DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_fn)



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




extractor_,dataloader = get_dataloader(ALLSCENES[scene_st:scene_st+scene_batch])
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
        scene_st += scene_batch
        if scene_st + scene_batch > trainScenes:
            scene_st = 0
        extractor_.close()
        extractor_, dataloader = get_dataloader(ALLSCENES[scene_st:scene_st + scene_batch])



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
