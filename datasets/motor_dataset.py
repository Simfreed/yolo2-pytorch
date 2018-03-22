import torch
import torchvision.transforms as transforms
from skimage import data, io
from torch.utils.data import Dataset
import numpy as np
myroot = '/project/weare-dinner/simonfreedman/cytomod/out/insulin_tracks/motor_walk100'

maxpixel = np.power(2,16)

class MotorImageDataset(Dataset):

    def __init__(self, root_dir, ni, nb, na, transform=None):

        self.frames_dir = '{0}/frames'.format(root_dir)
        self.truths_dir = '{0}/truths'.format(root_dir)
        self.transform  = transform
        self.n_images   = ni
        self.n_before   = nb
        self.n_after    = na

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        
        images = []
#        truths = []

        for t in np.arange(idx + 1 - self.n_before, idx + 2 + self.n_after):
            
            t2 = min(max(1, t), self.n_images - 1)
            images.append(io.imread('{0}/t{1}.tiff'.format(self.frames_dir, t2)))
#            truths.append(np.loadtxt('{0}/t{1}.csv'.format(self.truths_dir, t2), delimiter=','))
        truth = np.loadtxt('{0}/t{1}.csv'.format(self.truths_dir, idx + 1), delimiter=',')
        sample = {'image':np.array(images), 'truth':np.array([truth])}
#        sample = {'image':np.array(images), 'truth':np.array(truths)}
        #sample = {'image':np.array(images), 'truth':idx}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor2(object):
    def __call__(self, sample):
        image, truth = sample['image'], sample['truth']
        return torch.from_numpy(image/maxpixel).type(torch.FloatTensor), torch.from_numpy(truth).type(torch.FloatTensor)
#        return torch.from_numpy((image/maxpixel)).type(torch.FloatTensor), truth
