import os
import glob
import argparse
import cv2

import numpy as np

def make_dataset_for_srcnn(opt, div2k_dir, patch_dst_dir):
    
    path_opt = '_x' + str(opt.scale_factor) + 'lr' + str(opt.lr_img_size)
    train_lr_patches_dir = os.path.join(patch_dst_dir, 'train_patches' + path_opt, 'lr')
    train_hr_patches_dir = os.path.join(patch_dst_dir, 'train_patches' +path_opt, 'hr')
    valid_lr_patches_dir = os.path.join(patch_dst_dir, 'valid_patches' + path_opt, 'lr')
    valid_hr_patches_dir = os.path.join(patch_dst_dir, 'valid_patches' + path_opt, 'hr')

    print("train_lr_patches_dir: " + train_lr_patches_dir)
    print("train_hr_patches_dir: " + train_hr_patches_dir)
    print("valid_lr_patches_dir: " + valid_lr_patches_dir)
    print("valid_hr_patches_dir: " + valid_hr_patches_dir)

    if not os.path.exists(train_lr_patches_dir):
        os.makedirs(train_lr_patches_dir)
    if not os.path.exists(train_hr_patches_dir):
        os.makedirs(train_hr_patches_dir)

    if not os.path.exists(valid_lr_patches_dir):
        os.makedirs(valid_lr_patches_dir)
    if not os.path.exists(valid_hr_patches_dir):
        os.makedirs(valid_hr_patches_dir)

    scale_factor = opt.scale_factor
    lr_img_size = opt.lr_img_size
    hr_img_size = lr_img_size * scale_factor
    stride = opt.stride

    div2k_train_hr_dir = os.path.join(div2k_dir, 'DIV2K_train_HR')
    div2k_train_lr_dir = os.path.join(div2k_dir, 'DIV2K_train_LR_bicubic', 'X' + str(scale_factor))
    div2k_valid_hr_dir = os.path.join(div2k_dir, 'DIV2K_valid_HR')
    div2k_valid_lr_dir = os.path.join(div2k_dir, 'DIV2K_valid_LR_bicubic', 'X' + str(scale_factor))

    print(div2k_train_hr_dir)
    print(div2k_train_lr_dir)

    hr_train_list = os.listdir(div2k_train_hr_dir)
    lr_train_list = os.listdir(div2k_train_lr_dir)
    hr_valid_list = os.listdir(div2k_valid_hr_dir)
    lr_valid_list = os.listdir(div2k_valid_lr_dir)

    hr_train_list.sort()
    lr_train_list.sort()
    hr_valid_list.sort()
    lr_valid_list.sort()

    print(len(hr_train_list))
    print(len(lr_train_list))
    print(len(hr_valid_list))
    print(len(lr_valid_list))
    
    num_train_samples = len(hr_train_list)
    num_valid_samples = len(hr_valid_list)

    print('Number of training samples: ' + str(num_train_samples))
    print('Number of validation samples: ' + str(num_valid_samples))

    for i, (hr, lr) in enumerate(zip(hr_train_list, lr_train_list)):
        print("Processing training image LR {} and HR {}".format(lr, hr))
        lr_train_img_path = os.path.join(div2k_train_lr_dir, lr)
        hr_train_img_path = os.path.join(div2k_train_hr_dir, hr)
        lr_img = cv2.imread(lr_train_img_path)
        hr_img = cv2.imread(hr_train_img_path)

        lr_h, lr_w, _ = lr_img.shape
        hr_h, hr_w, _ = hr_img.shape

        lr_h = lr_h - np.mod(lr_h, lr_img_size)
        lr_w = lr_w - np.mod(lr_w, lr_img_size)
        hr_h = hr_h - np.mod(hr_h, hr_img_size)
        hr_w = hr_w - np.mod(hr_w, hr_img_size)
        
        for y in range(0, lr_h - lr_img_size + 1, stride):
            for x in range(0, lr_w - lr_img_size + 1, stride):
                lr_patch = lr_img[y:y+lr_img_size, x:x+lr_img_size, :]
                hr_patch = hr_img[y*scale_factor:y*scale_factor+hr_img_size, x*scale_factor:x*scale_factor+hr_img_size, :]

                lr_img_file = "%04d_%05d_%05d.png" % (i+1, x, y)
                lr_img_file = os.path.join(train_lr_patches_dir, lr_img_file)
                hr_img_file = "%04d_%05d_%05d.png" % (i+1, x, y)
                hr_img_file = os.path.join(train_hr_patches_dir, hr_img_file)

                cv2.imwrite(lr_img_file, lr_patch)
                cv2.imwrite(hr_img_file, hr_patch)

    for i, (hr, lr) in enumerate(zip(hr_valid_list, lr_valid_list)):
        print("Processing validation image LR {} and HR {}".format(lr, hr))
        lr_valid_img_path = os.path.join(div2k_valid_lr_dir, lr)
        hr_valid_img_path = os.path.join(div2k_valid_hr_dir, hr)
        lr_img = cv2.imread(lr_valid_img_path)
        hr_img = cv2.imread(hr_valid_img_path)

        lr_h, lr_w, _ = lr_img.shape
        hr_h, hr_w, _ = hr_img.shape

        lr_h = lr_h - np.mod(lr_h, lr_img_size)
        lr_w = lr_w - np.mod(lr_w, lr_img_size)
        hr_h = hr_h - np.mod(hr_h, hr_img_size)
        hr_w = hr_w - np.mod(hr_w, hr_img_size)

        for y in range(0, lr_h - lr_img_size + 1, stride):
            for x in range(0, lr_w - lr_img_size + 1, stride):
                lr_patch = lr_img[y:y+lr_img_size, x:x+lr_img_size, :]
                hr_patch = hr_img[y*scale_factor:y*scale_factor+hr_img_size, x*scale_factor:x*scale_factor+hr_img_size, :]

                lr_img_file = "lr_%04d_%05d_%05d.png" % (i+1, x, y)
                lr_img_file = os.path.join(valid_lr_patches_dir, lr_img_file)
                hr_img_file = "hr_%04d_%05d_%05d.png" % (i+1, x, y)
                hr_img_file = os.path.join(valid_hr_patches_dir, hr_img_file)

                cv2.imwrite(lr_img_file, lr_patch)
                cv2.imwrite(hr_img_file, hr_patch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_img_size', type=int, default=64,
                        help='Size of each low resolution image dimension')
    parser.add_argument('--stride', type=int, default=64,
                        help='Size of stride')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels')
    parser.add_argument('--scale_factor', type=int, default=2,
                        help='Scale factor from low resolution to super resolution image')
    
    # parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
    opt = parser.parse_args()
    print(opt)

    base_dir = os.getcwd()
    div2k_dir = os.path.join(base_dir, '../../../data/DIV2K_100')
    patch_dst_dir = os.path.join(div2k_dir, '../../../data/super-resolution/div2k_100')
    div2k_dir = os.path.abspath(div2k_dir)
    patch_dst_dir = os.path.abspath(patch_dst_dir)
    print(div2k_dir)
    print(patch_dst_dir)

    print("Create data for training with DIV2k dataset")
    make_dataset_for_srcnn(opt, div2k_dir, patch_dst_dir)