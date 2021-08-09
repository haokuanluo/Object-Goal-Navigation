import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_originalmodel():
    return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

def getMask(dic,truth = False):
    labels = dic['labels'].cpu().numpy()
    masks = dic['masks'].cpu().numpy()
    if not truth:
        scores = dic['scores'].cpu().numpy()
    ans = np.zeros((256,256),dtype = int)
    n = masks.shape[0]
    print(labels)
    for i in range(n):
        if (not truth) and scores[i] < 0.1:
            continue
        if truth:
            ans = ans * (1 - masks[i, :, :]) + labels[i] * masks[i, :, :]
        else:
            ans = ans * (1-masks[i,0,:,:]) + labels[i]*masks[i,0,:,:]
    return ans

def saveTruthMask(dic,save_path,batchno):
    def show_row(img, path,batchno,maskno):
        plt.figure(figsize=(12, 8))

        plt.imshow(img)


        fn = save_path+ 'truth_{}_{}.png'.format(
            batchno,maskno)
        plt.savefig(fn)
        plt.close()

    if not path.exists(save_path):
        os.makedirs(save_path)
    labels = dic['labels'].cpu().numpy()
    masks = dic['masks'].cpu().numpy()

    n = masks.shape[0]
    for i in range(n):
        show_row(masks[i,:,:],save_path,batchno,i)

def get_pred(model,batch):
    ans = {}
    device = torch.device('cuda')
    image,targets = batch
    ans['rgb'] = image
    truth = list(target['truth'] for target in targets)
    ground_truth = []
    i = 0
    for target in targets:
        mask = getMask(target,True)
        saveTruthMask(target,"semantic/images/gtmask/",i)
        i=i+1
        ground_truth.append(mask)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.int64)

    images = list(img.to(device) for img in image)
    print(images[0].shape)
    print(images[0])
    exit(0)
    outputs = model(images)

    predicts = []
    for dic in outputs:
        print('mask shape',dic['masks'].shape)
        mask = getMask(dic)
        predicts.append(mask)
    predicts = torch.as_tensor(predicts,dtype = torch.int64)
    ans['truth'] = truth
    ans['target'] = ground_truth
    ans['prediction'] = predicts
    return ans
