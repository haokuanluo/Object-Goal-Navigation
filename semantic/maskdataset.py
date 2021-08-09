import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from habitat_sim.utils.data import ImageExtractor

import pickle

def get_transform(train):
    trans = []
    trans.append(transforms.ToTensor())
    if train:
        trans.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(trans)

class MaskRCNNDataset(Dataset):
    def __init__(self, extractor, transforms=None):
        # Define an ImageExtractor
        self.extractor = extractor

        # We will perform preprocessing transforms on the data
        self.transforms = transforms

        # Habitat sim outputs instance id's from the semantic sensor (i.e. two
        # different chairs will be marked with different id's). So we need
        # to create a mapping from these instance id to the class labels we
        # want to predict. We will use the below dictionaries to define a
        # funtion that takes the raw output of the semantic sensor and creates
        # a 2d numpy array of out class labels.
        self.labels = {
            'background': 0,
            'chair': 1,
            'table': 2,
            'picture':3,
            'cabinet':4,
            'cushion':5,
            'sofa':6,
            'bed':7,
            'chest_of_drawers':8,
            'plant':9,
            'sink':10,
            'toilet':11,
            'stool':12,
            'towel':13,
            'tv_monitor':14,
            'shower':15,
            'bathtub':16,
            'counter':17,
            'fireplace':18,
            'gym_equipment':19,
            'seating':20,
            'clothes':21
        }
        self.instance_id_to_name = self.extractor.instance_id_to_name
        self.map_to_class_labels = np.vectorize(
            lambda x: self.labels.get(self.instance_id_to_name.get(x, 0), 0)
        )

    def __len__(self):
        return len(self.extractor)

    def isBackground(self,x):
        return self.getLabel(x) == 0

    def getLabel(self,x):
        return self.labels.get(self.instance_id_to_name.get(x, 0), 0)

    def __getitem__(self, idx):
        while True:
            sample = self.extractor[idx]
            img = sample['rgba'][:, :, :3]
            mask = sample['semantic']
            truth_mask = self.get_class_labels(mask)
            mask = np.array(mask)
            obj_ids = np.unique(mask)
            masks = mask == obj_ids[:, None, None]
            mask2 = []
            num_objs = len(obj_ids)
            boxes = []
            labels = []
            for i in range(num_objs):
                if self.isBackground(obj_ids[i]):
                    continue
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if (xmax - xmin) * (ymax - ymin) < 100:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.getLabel(obj_ids[i]))
                mask2.append(masks[i])
            masks = mask2
            # convert everything into a torch.Tensor
            num_objs = len(boxes)
            if num_objs != 0:
                break
            idx = idx+1
            if idx >= self.__len__():
                idx = 0


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels,dtype = torch.int64)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["truth"] = truth_mask.astype(int)

        if self.transforms is not None:
            img = self.transforms(img)
            target['truth'] = self.transforms(target['truth']).squeeze(0)

        return img, target








    def get_class_labels(self, raw_semantic_output):
        return self.map_to_class_labels(raw_semantic_output)


class MaskRCNNDatasetPresaved(Dataset):
    def __init__(self, filePath):



        self.data = pickle.load(open(filePath,"rb"))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]








def show_batch(sample_batch):
    def show_row(imgs, batch_size, img_type):
        plt.figure(figsize=(12, 8))
        for i, img in enumerate(imgs):
            ax = plt.subplot(1, batch_size, i + 1)
            ax.axis("off")
            if img_type == 'rgb':
                plt.imshow(img.numpy().transpose(1, 2, 0))
            elif img_type == 'truth':
                plt.imshow(img.numpy())

        print("saved")
        plt.savefig('semantic/images/eval4/dataset.png')
        plt.close()
    batch_size = len(sample_batch['rgb'])
    for k in sample_batch.keys():
        show_row(sample_batch[k], batch_size, k)


if __name__ == "__main__":
    print('here')
    SCENE_NAME = 'data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb'
    BATCH_SIZE = 4
    extractor = ImageExtractor(SCENE_NAME, img_size=(256, 256),
                               output=['rgba', 'semantic'])
    dataset = MaskRCNNDataset(extractor,
                                          transforms=transforms.Compose(
                                              [transforms.ToTensor()])
                                          )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    _, sample_batch = next(enumerate(dataloader))
    print('117')
    show_batch(sample_batch)
