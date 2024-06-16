import numpy as np
import torch
import os
import cv2
import math
import datetime
from models.superpoint import SuperPoint

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from models.utils import read_image_wxl

class SparseDatasetWxl(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, config):

        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]
        self.nfeatures = 1024
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'

        self.superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) 
        # ul_y,ul_x =640-100, 640-300  ############################
        # image = image[ul_y:ul_y+300,ul_x:ul_x+300] ######################
        height, width  = image.shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        # warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)
        warp = np.random.randint(-124,124, size=(4, 2)).astype(np.float32)

        # get the corresponding warped image
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0])) # return an image type
        
        # extract keypoints of the image pair using SIFT
        # kp1, descs1 = sift.detectAndCompute(image, None)
        # kp2, descs2 = sift.detectAndCompute(warped, None)
        
        # extract keypoints of the image pair using SuperPoint
        image0, inp0, scales0 = read_image_wxl( image, [-1], 0, True)
        image1, inp1, scales1 = read_image_wxl( warped, [-1], 0, True)
        data = {'image0': inp0, 'image1': inp1}
        pred = {}
        pred0 = self.superpoint({'image': data['image0']})
        pred = {**pred, **{k+'0': v for k, v in pred0.items()}}

        pred1 = self.superpoint({'image': data['image1']})
        pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        # pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        
        kp1_np = pred["keypoints0"][0].cpu().detach().numpy()
        kp2_np = pred["keypoints1"][0].cpu().detach().numpy()
        
        # limit the number of keypoints
        # kp1_num = min(self.nfeatures, len(kp1_np))
        # kp2_num = min(self.nfeatures, len(kp2_np))
        # kp1 = kp1[:kp1_num]
        # kp2 = kp2[:kp2_num]

        # kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        # kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])
        # kp1_np = kp1_np[:kp1_num]
        # kp2_np = kp2_np[:kp2_num]


        # skip this image pair if no keypoints detected in image
        if len(kp1_np) < 1 or len(kp2_np) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            } 

        # confidence of each key point
        # scores1_np = np.array([kp.response for kp in kp1]) 
        # scores2_np = np.array([kp.response for kp in kp2])

        # kp1_np = kp1_np[:kp1_num, :]
        # kp2_np = kp2_np[:kp2_num, :]
        # descs1 = descs1[:kp1_num, :]
        # descs2 = descs2[:kp2_num, :]

        # obtain the matching matrix of the image pair
        # matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] 
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2_np)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1_np)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        # descs1 = np.transpose(descs1 / 256.)
        # descs2 = np.transpose(descs2 / 256.)

        image = torch.from_numpy(image/255.).double()[None].cuda()
        warped = torch.from_numpy(warped/255.).double()[None].cuda()

        return{
            'keypoints0': pred["keypoints0"],
            'keypoints1': pred["keypoints1"],
            'descriptors0': pred["descriptors0"][0],
            'descriptors1': pred["descriptors1"][0],
            'scores0': pred["scores0"][0],
            'scores1': pred["scores1"][0],
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        } 
        
        # return{
        #     'keypoints0': list(pred["keypoints0"]),
        #     'keypoints1': list(pred["keypoints1"]),
        #     'descriptors0': list(pred["descriptors0"]),
        #     'descriptors1': list(pred["descriptors0"]),
        #     'scores0': list(pred["scores0"]),
        #     'scores1': list(pred["scores1"]),
        #     'image0': image,
        #     'image1': warped,
        #     'all_matches': list(all_matches),
        #     'file_name': file_name
        # } 

