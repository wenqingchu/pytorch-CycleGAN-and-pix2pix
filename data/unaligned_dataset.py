import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import cv2

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if self.root == './datasets/semanticlabels':
            A_paths = []
            A_dir = os.path.join(self.root, 'cityscapes_prediction/gtFine')
            A_list_path = 'semanticlabels_list/' + opt.phase + 'A.txt'
            A_img_ids = [i_id.strip() for i_id in open(A_list_path)]
            for name in A_img_ids:
                A_img_file = os.path.join(A_dir, "%s/%s" % (opt.phase, name[:-3] + "npy"))
                A_paths.append(A_img_file)
            self.A_paths = A_paths
            B_paths = []
            B_dir = os.path.join(self.root, 'GTA5_prediction/labels')
            B_list_path = 'semanticlabels_list/' + opt.phase + 'B.txt'
            B_img_ids = [i_id.strip() for i_id in open(B_list_path)]
            for name in B_img_ids:
                B_img_file = os.path.join(B_dir, "%s" % (name[:-3]+"npy"))
                B_paths.append(B_img_file)
            self.B_paths = B_paths
        elif opt.model == 'stn_prediction':
            A_paths = []
            A_dir = os.path.join(self.root, 'GTA5_prediction/labels')
            A_list_path = 'semanticlabels_list/trainB.txt'
            A_img_ids = [i_id.strip() for i_id in open(A_list_path)]
            for name in A_img_ids:
                A_img_file = os.path.join(A_dir, "%s" % (name[:-3] + "npy"))
                A_paths.append(A_img_file)
            self.A_paths = A_paths
            B_paths = []
            B_dir = os.path.join(self.root, 'GTA5_prediction/labels')
            B_list_path = 'semanticlabels_list/valB.txt'
            B_img_ids = [i_id.strip() for i_id in open(B_list_path)]
            for name in B_img_ids:
                B_img_file = os.path.join(B_dir, "%s" % (name[:-3] + "npy"))
                B_paths.append(B_img_file)
            self.B_paths = B_paths








        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        if self.root == './datasets/semanticlabels':
            # A is cityscapes, B is GTA5
            label2train = [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 0], [8, 1],
                           [9, 255], [10, 255], [11, 2], [12, 3],
                           [13, 4], [14, 255], [15, 255], [16, 255], [17, 5], [18, 255], [19, 6], [20, 7], [21, 8],
                           [22, 9], [23, 10], [24, 11], [25, 12],
                           [26, 13], [27, 14], [28, 15], [29, 255], [30, 255], [31, 16], [32, 17], [33, 18], [-1, 255]]

            #A_label = Image.open(A_path)
            #B_label = Image.open(B_path)
            #A_label = self.transform(A_label)
            #A_label = A_label.resize((256,256), Image.NEAREST)
            #B_label = self.transform(B_label)
            #B_label = B_label.resize((256,256), Image.NEAREST)
            A_label = np.load(A_path)
            B_label = np.load(B_path)
            A_label = cv2.resize(A_label, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            B_label = cv2.resize(B_label, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

            A_label = np.asarray(A_label, np.float32)
            #A_label = A_label[np.newaxis, :]
            A_label = np.transpose(A_label, (2,0,1))
            B_label = np.asarray(B_label, np.float32)
            #B_label = B_label[np.newaxis, :]
            B_label = np.transpose(B_label, (2,0,1))

            #A_label_copy = 255 * np.ones(A_label.shape, dtype=np.float32)
            #for ind in range(len(label2train)):
            #    A_label_copy[A_label == label2train[ind][0]] = label2train[ind][1]
            #A_label_copy[A_label_copy == 255] = 19
            #A_label_copy = A_label_copy
            #B_label_copy = B_label
            #B_label_copy[B_label_copy == 255] = 19
            A_label_copy = A_label
            B_label_copy = B_label
            return {'A': A_label_copy, 'B': B_label_copy,
                    'A_paths': A_path, 'B_paths': B_path}

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
