import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import cv2
import torchvision.transforms as transform

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.root_gta5 = './datasets/da_stn/GTA5'
        self.root_cityscapes = './datasets/da_stn/Cityscapes/data'
        self.list_path_gta5 = './datasets/da_stn/gta5_list/train.txt'
        self.list_path_cityscapes = './datasets/da_stn/cityscapes_list/train.txt'
        self.transform_gta5 = transform.Compose([transform.ToTensor(),transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.transform_cityscapes = transform.Compose([transform.ToTensor(),transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.img_ids_gta5 = [i_id.strip() for i_id in open(self.list_path_gta5)]
        self.img_ids_cityscapes = [i_id.strip() for i_id in open(self.list_path_cityscapes)]
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        self.crop_size_gta5 = (1280, 720)
        self.crop_size_cityscapes = (1024, 512)

        if opt.model == 'da_stn':
            A_paths = []
            C_paths = []
            for name in self.img_ids_gta5:
                A_img_file = os.path.join(self.root_gta5, "images/%s" % name)
                A_label_file = os.path.join(self.root_gta5, "labels/%s" % name)
                A_paths.append(A_img_file)
                C_paths.append(A_label_file)


            B_paths = []
            for name in self.img_ids_cityscapes:
                B_img_file = os.path.join(self.root_cityscapes, "leftImg8bit/train/%s" % name)
                B_paths.append(B_img_file)


            self.C_paths = C_paths
            self.B_paths = B_paths
            self.A_paths = A_paths

        else:
            raise NotImplementedError('Model name [%s] is not recognized' % opt.model)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        C_path = self.C_paths[index % self.C_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_image = Image.open(A_path).convert('RGB')
        A_label = Image.open(C_path)

        A_image = A_image.resize(self.crop_size_gta5, Image.BICUBIC)
        A_label = A_label.resize(self.crop_size_gta5, Image.NEAREST)
        A_label = np.asarray(A_label, np.float32)

        B_image = Image.open(B_path).convert('RGB')
        B_image = B_image.resize(self.crop_size_cityscapes, Image.BICUBIC)

        if self.transform_gta5 is not None:
            A_image = self.transform_gta5(A_image)
            B_image = self.transform_cityscapes(B_image)
        else:
            A_image = np.asarray(A_image, np.float32)
            A_image = A_image[:, :, ::-1]  # change to BGR
            A_image -= self.mean
            A_image = A_image.transpose((2, 0, 1))

            B_image = np.asarray(B_image, np.float32)
            B_image = B_image[:, :, ::-1]  # change to BGR
            B_image -= self.mean
            B_image = B_image.transpose((2, 0, 1))

        if self.opt.model == 'da_stn':
            return {'A': A_image, 'B': B_image, 'C': A_label,
                    'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}

        else:
            raise NotImplementedError('Model name [%s] is not recognized' % self.opt.model)

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
