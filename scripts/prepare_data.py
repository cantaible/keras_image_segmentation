import os
import cv2 as cv
import shutil
import numpy as np
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
before_image_dir = '/home/tju/dzm/Segmentation/dataset/DeepCrack_224/train_img'
before_anno_dir = '/home/tju/dzm/Segmentation/dataset/DeepCrack_224/train_lab'
before_test_image_dir = '/home/tju/dzm/Segmentation/dataset/DeepCrack_224/test_img'
before_test_anno_dir = '/home/tju/dzm/Segmentation/dataset/DeepCrack_224/test_lab'
after_root_dir = os.path.join(root_dir, 'dataset')
after_test_image_dir = os.path.join(after_root_dir, 'val_image')
after_test_anno_dir = os.path.join(after_root_dir, 'val_anno')
after_fully_image_dir = os.path.join(after_root_dir, 'fully_image')
after_fully_anno_dir = os.path.join(after_root_dir, 'fully_anno')
if os.path.exists(after_root_dir):
    shutil.rmtree(after_root_dir)
os.mkdir(after_root_dir)
os.mkdir(after_test_image_dir)
os.mkdir(after_test_anno_dir)
os.mkdir(after_fully_image_dir)
os.mkdir(after_fully_anno_dir)



def process_FSL_dataset():
    anno_list = os.listdir(before_anno_dir)
    for anno_name in tqdm(anno_list):
        anno_file_path = os.path.join(before_anno_dir, anno_name)
        anno = cv.imread(anno_file_path)
        # (224, 224, 3)
        anno = anno / 255
        anno = anno.astype('uint8')
        assert np.unique(anno).size == 2
        output_anno_name = os.path.join(after_fully_anno_dir, anno_name)
        cv.imwrite(output_anno_name, anno)
        shutil.copyfile(os.path.join(before_image_dir, anno_name[:-3] + 'jpg'),
                        os.path.join(after_fully_image_dir, anno_name[:-3] + 'jpg'))


def process_val_dataset():
    anno_list = os.listdir(before_test_anno_dir)
    for anno_name in tqdm(anno_list):
        anno_file_path = os.path.join(before_test_anno_dir, anno_name)
        anno = cv.imread(anno_file_path)
        # (224, 224, 3)
        anno = anno / 255
        anno = anno.astype('uint8')
        assert np.unique(anno).size == 2
        output_anno_name = os.path.join(after_test_anno_dir, anno_name)
        cv.imwrite(output_anno_name, anno)
        shutil.copyfile(os.path.join(before_test_image_dir, anno_name[:-3] + 'jpg'),
                        os.path.join(after_test_image_dir, anno_name[:-3] + 'jpg'))

process_FSL_dataset()
# process_WSL_dataset()
process_val_dataset()