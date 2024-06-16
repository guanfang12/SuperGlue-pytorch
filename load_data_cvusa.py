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

# import get_BEV, torch 等
import matplotlib.pyplot as plt
import time
import cv2 as cv
from scipy.spatial.transform import Rotation

# 读入匹配网络
from match_pairs import *
from models.utils import read_image_wxl
import sys
import glob

import matplotlib
def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def get_BEV_tensor(img,Ho, Wo, Fov = 170, dty = -20, dx = 0, dy = 0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t0 = time.time()

    if len(img.shape) ==3 :
        Hp, Wp , _ = img.shape                                # 全景图尺寸
    else:
        Hp, Wp  = img.shape                                # 全景图尺寸
    if dty != 0 or Wp != 2*Hp:
        ty = (Wp/2-Hp)/2 + dty                                                 # 非标准全景图补全
        matrix_K = np.array([[1,0,0],[0,1,ty],[0,0,1]])
        img = cv.warpPerspective(img,matrix_K,(int(Wp),int(Hp+(Wp/2-Hp))))
    ######################
    t1 = time.time()
    frame = torch.from_numpy(img).to(device)  
    t2 = time.time()

    if len(frame.shape) ==3 :
        Hp, Wp , _ = frame.shape                                # 全景图尺寸
    else:
        Hp, Wp  = frame.shape                                # 全景图尺寸
    # Wp, Hp = 16384, 8192                                # 全景图尺寸
    Fov = 170 * torch.pi / 180                               # 视场角
    center = torch.tensor([Wp/2+dx,Hp+dy]).to(device)                  # 俯瞰图中心
    # Ho, Wo =  500,500                                        # 俯瞰图尺寸

    anglez = center[0] * 2 * torch.pi / Wp
    angley = torch.pi / 2 - center[1] * torch.pi / Hp

    f = Wo/2/torch.tan(torch.tensor(Fov/2))
    r = Rotation.from_euler('zy',[anglez.cpu(),angley.cpu()], degrees=False)
    R02 = torch.from_numpy(r.as_matrix()).float().to(device)
    out = torch.zeros((Wo, Ho,2)).to(device)
    f0 = torch.zeros((Wo, Ho,3)).to(device)  
    f0[:,:,0] = torch.ones((Wo, Ho)).to(device) *f
    f0[:,:,1] = -Wo/2 + torch.ones((Ho, Wo)).to(device)  *torch.arange(Wo).to(device)  
    f0[:,:,2] = -Ho/2 + (torch.ones((Ho, Wo)).to(device)  *(torch.arange(Ho)).to(device)).T
    f1 = R02@ f0.reshape((-1,3)).T  # x,y,z (3*N)
    f1_0 = torch.sqrt(torch.sum(f1**2,0))
    f1_1 = torch.sqrt(torch.sum(f1[:2,:]**2,0))
    theta = torch.arccos(f1[2,:]/f1_0)
    phi = torch.arccos(f1[0,:]/f1_1)
    mask = f1[1,:] <  0 
    phi[mask] = 2 * torch.pi - phi[mask]
    #################################
    phi = 2 * torch.pi - phi+ torch.pi
    mask = phi >  2 * torch.pi 
    phi[mask] = phi[mask] - 2 * torch.pi 
    #################################
    i_p = theta  / torch.pi * Hp
    j_p = phi  / (2 * torch.pi) * Wp
    out[:,:,0] = i_p.reshape((Ho, Wo))
    out[:,:,1] = j_p.reshape((Ho, Wo))
    t3 = time.time()


    src0 = torch.floor(out).int().to(device)
    src1 = src0 + torch.ones(src0.shape, dtype= int).to(device)

    mask = src0[:,:,0] >=  Hp
    src0[:,:,0][mask] = ((torch.ones(out.shape, dtype= int)[:,:,0]*(Hp-1)).to(device)[mask]).int()
    mask = src0[:,:,1] >= Wp
    src0[:,:,1][mask] = ((torch.ones(out.shape, dtype= int)[:,:,0]*(Wp-1)).to(device)[mask]).int()

    mask = src1[:,:,0] >=  Hp
    src1[:,:,0][mask] = (torch.ones(out.shape, dtype= int)[:,:,0]*(Hp-1)).to(device)[mask]
    mask = src1[:,:,1] >= Wp
    src1[:,:,1][mask] = (torch.ones(out.shape, dtype= int)[:,:,0]*(Wp-1)).to(device)[mask]

    d = out - src0
    w0 = ((1 - d[:,:,1])*(1 - d[:,:,0])).reshape((Ho,Wo,1))
    w1 = (d[:,:,1]*(1 - d[:,:,0])).reshape((Ho,Wo,1))
    w2 = ((1 - d[:,:,1])*d[:,:,0]).reshape((Ho,Wo,1))
    w3 = (d[:,:,1]*d[:,:,0]).reshape((Ho,Wo,1))

    # BEV[Ho - 1 - i,j] = w0 * img[srcV0,srcU0] + w1 * img[srcV0,srcU1] + w2 * img[srcV1,srcU0] + w3 * img[srcV1,srcU1]
    BEV =w0*frame[src0[:,:,0].long(),src0[:,:,1].long()] + w1*frame[src0[:,:,0].long(),src1[:,:,1].long()]+w2*frame[src1[:,:,0].long(),src0[:,:,1].long()]+ w3*frame[src1[:,:,0].long(),src1[:,:,1].long()]

    t4  = time.time()
    # plt.imshow(BEV.cpu().int()[:,:,[2,1,0]])
    # plt.imshow(img[out.astype(int)[:,:,0],out.astype(int)[:,:,1]].astype(int)[:,:,[2,1,0]])
    # print("Read image ues {:.2f} ms, warpPerspective image use {:.2f} ms, Get matrix ues {:.2f} ms, Get out ues {:.2f} ms, All out ues {:.2f} ms.".format((t1-t0)*1000,(t2-t1)*1000, (t3-t2)*1000,(t4-t3)*1000,(t4-t0)*1000))
    return BEV.cpu().int()

class SparseDatasetWxl(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, config):

        self.root = '/home/wxl/Data'
        self.files = []
        # self.files += [train_path + f for f in os.listdir(train_path)]
        self.nfeatures = 1024
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        
        # 设置相关参数
        parser = set_pares()    
        # opt = parser.parse_args(args=['--superglue', 'outdoor','--resize','640','640'])
        opt = parser.parse_args(args=['--superglue', 'outdoor'])
        # with open(opt.input_pairs, 'r') as f:
        #     pairs = [l.split() for l in f.readlines()]

        # 加载模型
        device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(device)
        self.superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
        
        # binmap_path = f"/home/wxl/Data/CVUSA/bingmap/20/*.jpg"
        binmap_path = os.path.join(self.root,"CVUSA/bingmap/20/*.jpg")
        self.sat_20_list = sorted(glob.glob(binmap_path))
        self.files = self.sat_20_list 

    def __len__(self):
        return len(self.files)
    
    def get_data_cvusa(self, idx):
        id_20 = int(self.sat_20_list[idx].split('/')[-1].split('.')[0])       
        # img_path0 = "/data/wxl/Data/CVUSA/bingmap/20/{:07}.jpg".format(id_20)
        # img_path1 = '/data/wxl/Data/CVUSA/streetview/panos/{:07}.jpg'.format(id_20)
        img_path0 = os.path.join(self.root,"CVUSA/bingmap/20/{:07}.jpg".format(id_20))
        img_path1 = os.path.join(self.root,'CVUSA/streetview/panos/{:07}.jpg'.format(id_20))
        sat = cv2.imread(img_path0,0)
        img1 = cv2.imread(img_path1,1)
        return sat,img1

    def __getitem__(self, idx):
        file_name = self.files[idx]
        # image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) 
        
        # 读入卫星图与对应全景图
        sat, img = self.get_data_cvusa(idx)
        BEV = get_BEV_tensor(img,500,500).numpy().astype(np.uint8)
        BEV = cv2.cvtColor(BEV,cv2.COLOR_BGR2GRAY)
        rot0, rot1 = 0, 0 # 正数逆时针旋转，负数顺时针 单位为 90 degree 
        start = time.time()
        with torch.no_grad():
            image0, inp0, scales0 = read_image_wxl(sat, [500], rot0, True)
            # image1, inp1, scales1 = read_image( img_path1, device, opt.resize, rot1, opt.resize_float)
            image1, inp1, scales1 = read_image_wxl(BEV, [-1], rot1, True)
            pred = self.matching({'image0': inp0, 'image1': inp1})
            # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'][0].cpu().numpy(), pred['keypoints1'][0].cpu().numpy()
            matches, conf = pred['matches0'][0].cpu().numpy(), pred['matching_scores0'][0].cpu().numpy()
        # print("Network use {:.2f} ms".format((time.time() - start)*1000))
        # print(len(matches),len(kpts0),len(kpts1),len(conf[conf > 0.2]))

        top_k = len(conf[conf > 0.2])                                 # 获得所有置信度大于阈值的匹配数量
        top_idx = np.argsort(conf)[-top_k:]                   # 获得所有置信度大于阈值的匹配索引
        match_kpts0 = kpts0[top_idx]                             # 获得合格匹配图一特征点索引
        match_kpts1 = kpts1[matches[top_idx]]        # 获得合格匹配图二特征点索引
        
        rot,num_correct = 4,0
        if len(conf[conf > 0.2])  >= 3:
            matAffine,mask = cv2.estimateAffinePartial2D(match_kpts1, match_kpts0, ransacReprojThreshold= 3) # , ransacReprojThreshold= 3
            H_matAffine = np.zeros((3,3))
            H_matAffine[:2,:3] = matAffine
            H_matAffine[2,2] = 1
            H = H_matAffine
            num_correct = np.sum(mask)
            U, Sigma, V = np.linalg.svd(matAffine[:2, :2])
            R = np.dot(U, V)
            rot = np.arccos(R[0][0])
            rot = rot/np.pi*180            
            # print(num_correct, rot )   # 
            

            # get the corresponding warped image
            # M = np.linalg.inv(H_matAffine)
            M = H_matAffine
        
        # extract keypoints of the image pair using SuperPoint
        data = {'image0': inp0, 'image1': inp1}
        pred = {}
        pred0 = self.superpoint({'image': data['image0']})
        pred = {**pred, **{k+'0': v for k, v in pred0.items()}}

        pred1 = self.superpoint({'image': data['image1']})
        pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        
        kp1_np = pred["keypoints0"][0].cpu().detach().numpy()
        kp2_np = pred["keypoints1"][0].cpu().detach().numpy()

        # skip this image pair if no keypoints detected in image
        if len(kp1_np) < 1 or len(kp2_np) < 1 or abs(rot)>3 or num_correct < 4:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image0,
                'image1': image1,
                'file_name': file_name
            } 

        warped = cv2.warpPerspective(src=image1, M=M, dsize=(image1.shape[1], image1.shape[0])) # return an image type
        
        # extract keypoints of the image pair using SIFT
        # kp1, descs1 = sift.detectAndCompute(image, None)
        # kp2, descs2 = sift.detectAndCompute(warped, None)
        
        # extract keypoints of the image pair using SuperPoint
        image0, inp0, scales0 = read_image_wxl( image1, [-1], 0, True)
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

        image = torch.from_numpy(image0/255.).double()[None].cuda()
        warped = torch.from_numpy(image1/255.).double()[None].cuda()

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

