import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        # for real images
        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))
        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]
        # for images with edge smooth
        edge_path = os.path.join(opt.dataroot + 'edge*')
        self.edge_dirs = sorted(glob.glob(edge_path))
        self.edge_paths = [sorted(make_dataset(d)) for d in self.edge_dirs]
        self.edge_sizes = [len(p) for p in self.edge_paths]

        self.batch_size = opt.batchSize
        self.number = 0
        self.DA = 0
        self.DB = 0
        self.DC = 0

        print('load data from:')
        for i in range(len(self.dirs)):
            print(self.dirs[i])
        for i in range(len(self.edge_dirs)):
            print(self.edge_dirs[i])

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    # for images without edges
    def load_edge_image(self, dom, idx):
        path = self.edge_paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain: # for test 
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_real = index
            else:
                real = 0
                index_real = random.randint(0,self.sizes[real]-1)
        else: # for train
            real = 0
            index_real = random.randint(0, self.sizes[real]-1)

        real = 0
        real_img, real_path = self.load_image(real, index_real) # real image
        bundle = {'real': real_img, 'path_real': real_path}
        

        if self.opt.isTrain:
            if (self.number % self.batch_size) == 0:
                self.DA, self.DB, self.DC = random.sample(range(len(self.dirs)-1), 3)
                self.number = self.number + 1
            else:
                self.number = self.number + 1
            index_A = random.randint(0, self.sizes[self.DA + 1] - 1)
            A_img, A_path = self.load_image(self.DA+1, index_A) # cartoon images of style 1
            bundle.update({'A': A_img, 'DA': self.DA, 'path_A': A_path} )

            index_B = random.randint(0, self.sizes[self.DB + 1] - 1) # cartoon images of style 2
            B_img, B_path = self.load_image(self.DB+1, index_B)
            bundle.update( {'B': B_img, 'DB': self.DB, 'path_B': B_path} ) # cartoon images of style 3

            index_C = random.randint(0,self.sizes[self.DC +1] - 1)
            C_img, C_path = self.load_image(self.DC+1, index_C)
            bundle.update({'C':C_img, 'DC':self.DC, 'path_C':C_path})

            # for images without edges
            edge_index_A = random.randint(0, self.edge_sizes[self.DA]-1)
            edge_A_img, edge_A_path = self.load_edge_image(self.DA, edge_index_A)
            bundle.update({'edge_A':edge_A_img, 'edge_path_A':edge_A_path} )

            edge_index_B = random.randint(0, self.edge_sizes[self.DB]-1)
            edge_B_img, edge_B_path = self.load_edge_image(self.DB, edge_index_B)
            bundle.update({'edge_B':edge_B_img, 'edge_path_B':edge_B_path})

            edge_index_C = random.randint(0, self.edge_sizes[self.DC] -1)
            edge_C_img, edge_C_path = self.load_edge_image(self.DC, edge_index_C)
            bundle.update({'edge_C':edge_C_img, 'edge_path_C':edge_C_path})

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)
        

    def name(self):
        return 'UnalignedDataset'

    
