import os
import cv2
import numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from tqdm import tqdm
import keras_segmentation as KS
from sklearn.metrics import roc_curve, auc
from segment_metrics import cal_pixel_accuracy, cal_mean_IoU, cal_mean_pixel_accuracy, get_statistics_binary_cls
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


model_name = 'vgg_unet'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
eval_images_path = os.path.join(root_dir, 'dataset', 'val_image')
eval_annotations_path = os.path.join(root_dir, 'dataset', 'val_anno')

count = 0

logs_folder_name = model_name + '_%d' % count
logs_folder_path = os.path.join(root_dir, 'logs', logs_folder_name)
checkpoints_path = os.path.join(logs_folder_path, 'checkpoint')
pred_folder_path = os.path.join(logs_folder_path, 'pred_img')
if not os.path.exists(pred_folder_path):
    os.mkdir(pred_folder_path)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)
# model = keras_segmentation.models.unet.vgg_unet(n_classes=2, input_height=224, input_width=224)
# model = select_models(model_name, n_classes=2, input_height=224, input_width=224)
model = KS.predict.model_from_checkpoint_path(checkpoints_path)
image_path_list = os.listdir(eval_images_path)
image_path_list.sort()
anno_path_list = os.listdir(eval_annotations_path)
anno_path_list.sort()
pred_list = []
gt_list = []
pred_flatten_list = []
gt_flatten_list = []


for inp, ann in tqdm(zip(image_path_list, anno_path_list)):
    pr = KS.predict.predict(model, os.path.join(eval_images_path, inp))
    # cv2.imwrite(os.path.join(pred_folder_path, inp), pr * 255)
    # (224, 224)
    gt = cv2.imread(os.path.join(eval_annotations_path, ann))[:, :, 0]
    gt =cv2.resize(gt, (pr.shape[0], pr.shape[1]))
    # (224, 224)
    # cv2.imwrite(os.path.join(pred_folder_path, inp), gt * 255)
    pred_flatten_list.append(np.reshape(pr, -1))
    gt_flatten_list.append(np.reshape(gt, -1))
pred_flatten = np.concatenate(pred_flatten_list)
gt_flatten = np.concatenate(gt_flatten_list)
PA = cal_pixel_accuracy(pred_flatten, gt_flatten)
MPA = cal_mean_pixel_accuracy(pred_flatten, gt_flatten, num_cls=2)
MIoU = cal_mean_IoU(pred_flatten, gt_flatten, num_cls=2)
Precision, Recall, F1 = get_statistics_binary_cls(pred_flatten, gt_flatten)
fpr, tpr, thresholds = roc_curve(gt_flatten, pred_flatten, pos_label=1)
print('PA:', PA)
print('MPA:', MPA)
print('MIoU:', MIoU)
print('Precision:', Precision)
print('Recall:', Recall)
print('F1:', F1)
print('auc:', auc(fpr, tpr))
