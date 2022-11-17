import os
from data.base_dataset import BaseDataset, get_transform, get_transform_no_data_aug
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import cv2
import numpy as np

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_grayscale = get_transform_no_data_aug(self.opt, grayscale=True)

    def reset_random_seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        random.seed(random_seed)
        np.random.seed(random_seed)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)#.convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # print(B_path)
        
        # B is H&E and there is no need for random consistency

        seed_A = np.random.randint(2147483647)
        seed_B = np.random.randint(2147483647)

        transform_B_fix_seed = get_transform(self.opt, grayscale=False, curImgIsHE = True) # get custom
        self.reset_random_seed(seed_B)
        B = transform_B_fix_seed(B_img)
        # self.reset_random_seed(seed_B)
        # B1 = transform_B_fix_seed(B_img)
        # # print('##########')
        # # # print(B.shape)

        # s_1 = np.random.randint(2147483647)
        # # random.seed(s_1)
        # # torch.manual_seed(s_1)
        # # transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        # # B1 = transform_B(B_img)
  
        # xxx = torch.eq(B, B1)
        # print(torch.unique(xxx))
        # print('##########')

        # deld = B.cpu().float().numpy()
        # deld = (np.transpose(deld, (1, 2, 0)) + 1) / 2.0 * 255.0
        # # print(deld.shape)
        # B_arr = np.asarray(B_img)
        # img_s  = []
        # img_s.append(B_arr)
        # img_s.append(deld)
        # img_stack = np.vstack(img_s)
        # cv2.imwrite('%s_before.png' % s_1,B_arr)
        # cv2.imwrite('%s_after.png' % s_1,deld)
        # # B_img.save("%s_before.png" % s_1)

        # cv2.imwrite("%s.png" % s_1,img_stack)
        # image_pil = Image.fromarray(deld)
        # image_pil.save("%s.png" % s_1)

        # apply image transformation
        #A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        
        transform_A_fix_seed = get_transform(self.opt, grayscale=True) # get custom

        X = []
        total_marker = 4 # for AF related only 27
        #contain_list = [0,2,11,12,15,17,18,20,22,24,26] #remove muc2
        #contain_list = [0,2,11,12,15,18,20,22,24,26]
        contain_list = [1]
        for i in range(0, total_marker):
            # select only 11 markers
            if i in contain_list:
                A_tmp = A_img.crop((i * 256, 0, i * 256 + 256, 256))
                self.reset_random_seed(seed_A)
                A_tmp = transform_A_fix_seed(A_tmp)
                
                # A_tmp1 = transform_A_fix_seed(A_tmp)
                # self.reset_random_seed(seed_A)
                # A_tmp2 = transform_A_fix_seed(A_tmp)

                # xxx = torch.eq(A_tmp1, A_tmp2)
                # print(torch.unique(xxx))

                # print()
                X.append(A_tmp)
        
        A_tensor = None
        # for i in range(29):
        ####################for i in range(0, total_marker):  #27):
        for i in range(0, len(contain_list)):
            if A_tensor is None:
                A_tensor = X[i]
            else:
                A_tensor = torch.cat((A_tensor, X[i]), 0)
        return {'A': A_tensor, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
