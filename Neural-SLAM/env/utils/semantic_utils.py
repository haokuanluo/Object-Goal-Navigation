import torchvision
import torch
import numpy as np
import math
def get_label_mapping():
    label_mapping = {
        62:0,
        63:1,
        64:2,
        65:3,
        70:4,
        72:5,
        67:6,
        79:7,
        81:8,
        82:9,
        84:10,
        85:11,
        86:12,
        47:13,
        44:14
    }
    return label_mapping

label_map = get_label_mapping()
interested = ['chair','couch','potted plant','bed','toilet','tv','dining table','oven','sink','refrigerator',
                  'book','clock','vase','cup','bottle']

valid = {
    'chair':0,
    'sofa':1,
    'plant':2,
    'bed':3,
    'toilet':4,
    'tv_monitor':5
}

valid_map_0_20 = {
    0: 'chair',
    5: 'sofa',
    8: 'plant',
    6: 'bed',
    10: 'toilet',
    13: 'tv_monitor'
}

def num_object():
    return len(valid_map_0_20.keys())

def convert_obj(obj):
    obj = obj[0]
    if obj in valid_map_0_20:
        return valid[valid_map_0_20[obj]]
    else:
        return None
def id2label(id):
    return interested[id]

def get_maskrcnn_model():
    return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

def getMask(dic):
    labels = dic['labels'].cpu().numpy()
    masks = dic['masks'].cpu().numpy()
    scores = dic['scores'].cpu().numpy()
    ans = np.zeros((256,256),dtype = int)
    n = masks.shape[0]
    for i in range(n):
        if scores[i] < 0.5:
            continue
        ans = ans * (1-masks[i,0,:,:]) + labels[i]*masks[i,0,:,:]
    return ans,dic

def remap(dic):
    l = []
    m = []
    s = []
    labels = dic['labels'].cpu().numpy()
    masks = dic['masks'].cpu().numpy()
    scores = dic['scores'].cpu().numpy()
    n = masks.shape[0]
    for i in range(n):
        if scores[i] < 0.3:
            continue
        if labels[i] in label_map:
            l.append(label_map[labels[i]])
            m.append(masks[i])
            s.append(scores[i])
        #else:
        #    l.append(labels[i])
        #    m.append(masks[i])
    ans = {
        'labels':np.array(l),
        'masks':np.array(m),
        'scores':np.array(s)
    }
    return ans
device = torch.device("cuda:0")
def get_semantic_mask(image,model):
    image = image/256.0
    image = np.array([image])
    image = torch.from_numpy(image).float().to(device)

    with torch.no_grad():
        output = model(image)[0]

    return remap(output)
    #return getMask(output)

x1=0
x2=0
y1=0
y2=0



def floodfill(vis,map,r,c):
    global x1
    global x2
    global y1
    global y2
    if vis[r][c] != 0:
        return
    if map[r][c] == 0:
        return
    vis[r][c] = 1
    x1 = min(x1,r)
    x2 = max(x2,r)
    y1 = min(y1,c)
    y2 = max(y2,c)
    for i in range(-1,1):
        for j in range(-1,1):
            floodfill(vis,map,r+i,c+j)


def get_boundary(vis,map,r,c):
    global x1
    global x2
    global y1
    global y2
    x1 = r
    x2 = r
    y1 = c
    y2 = c
    floodfill(vis,map,r,c)
    return x1,x2,y1,y2
