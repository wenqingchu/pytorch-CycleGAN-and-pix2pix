import torch
import numpy as np
from PIL import Image
import random
import torch.nn.functional as F
import cv2

theta = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float)
img_path = 'tmp4/wenqing_data/gta5_test_ori/'
save_path = 'tmp4/wenqing_data/gta5_test/'
for i in range(1000):
    print(i)
    single_img_path = img_path + str(i+4000+1).zfill(5) + '.npy'
    save_img_path = save_path + str(i+4000+1).zfill(5) + '.npy'
    save_label_path = save_path + str(i+4000+1).zfill(5) + '_label.npy'
    A_label = np.load(single_img_path)
    #A_label = cv2.resize(A_label, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    A_label = np.asarray(A_label, np.float32)
    A_label = np.transpose(A_label, (2,0,1))
    real_A = torch.from_numpy(A_label[np.newaxis,:])
    input_label = torch.nn.functional.softmax(real_A, dim=1)
    real_A = input_label.cuda()
    tmp_theta = np.zeros_like(theta)
    tmp_theta.dtype = theta.dtype
    for ii in range(2):
        for j in range(3):
            tmp = random.random()-1
            tmp_theta[ii][j] = theta[ii][j] + tmp*0.4
    tmp_theta[2][2] = 1
    theta_m = np.mat(tmp_theta)
    theta_i = np.asarray(theta_m.I)
    theta_i = theta_i[:2][:]
    tmp_theta = tmp_theta[:2][:]
    tmp_theta = tmp_theta[np.newaxis, :]
    tmp_theta = tmp_theta.repeat(real_A.size(0), axis=0)
    torch_theta = torch.from_numpy(tmp_theta)
    torch_theta = torch_theta.view(-1, 2, 3)
    grid = F.affine_grid(torch_theta, real_A.size())
    grid = grid.float()
    transformed_A = F.grid_sample(real_A, grid.cuda(), padding_mode='border')
    transformed_A = transformed_A.data.cpu().numpy()
    transformed_A = transformed_A[0]
    transformed_A = np.transpose(transformed_A, (1,2,0))
    np.save(save_img_path, transformed_A)
    np.save(save_label_path, theta_i)



