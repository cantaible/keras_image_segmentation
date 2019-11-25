import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
# from data_io import imread
from sklearn.metrics import roc_curve, auc


def get_statistics_binary_cls(pred, gt):
    pred = (pred > 0.5).astype(np.int32)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt != 1))
    fn = np.sum((pred != 1) & (gt == 1))
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = 2 * tp / (2 * tp + fp + fn)
    return Precision, Recall, F1


def cal_pixel_accuracy(pred, gt):
    pred = (pred > 0.5).astype(np.int32)
    true = (pred == gt).astype(np.int32)
    return true.sum() / pred.shape[0]


def cal_mean_pixel_accuracy(pred, gt, num_cls=2):
    pred = (pred > 0.5).astype(np.int32)
    category_pixel_accuracy = []
    for i in range(num_cls):
        true_num = ((pred == i) & (gt == i)).sum()
        belong_to_i_num = (gt == i).sum()
        category_pixel_accuracy.append(true_num / belong_to_i_num)
    return np.mean(category_pixel_accuracy)


def cal_mean_IoU(pred, gt, num_cls=2):
    pred = (pred > 0.5).astype(np.int32)
    category_mean_IoU = []
    for i in range(num_cls):
        Intersection = ((pred == i) & (gt == i)).sum()
        Union = (gt == i).sum() + (pred == i).sum() - Intersection
        category_mean_IoU.append(Intersection / Union)
    return np.mean(category_mean_IoU)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root_path = os.path.join(root_dir, 'segmentation', 'pred_img')
    img_folder_path = os.path.join(data_root_path, 'in')
    lab_folder_path = os.path.join(data_root_path, 'lab')
    pred_folder_path = os.path.join(data_root_path, 'prob_npy')
    file_list = os.listdir(img_folder_path)
    pred_list = []
    gt_list = []
    pred_flatten_list = []
    gt_flatten_list = []
    for file in file_list:
        lab_path = os.path.join(lab_folder_path, file)
        prob_path = os.path.join(pred_folder_path, file[:-3] + 'npy')
        lab = cv.imread(lab_path, 0)
        _, lab = cv.threshold(lab, 127, 255, cv.THRESH_BINARY)
        lab = lab / 255
        # lab.shape = [224, 224]
        assert np.unique(lab).size == 2
        prob = np.load(prob_path)
        prob = prob[0, :, :, 1]
        # prob.shape = [224, 224]
        pred_list.append(prob)
        gt_list.append(lab)
        pred_flatten_list.append(np.reshape(prob, -1))
        gt_flatten_list.append(np.reshape(lab, -1))
    pred_flatten = np.concatenate(pred_flatten_list)
    gt_flatten = np.concatenate(gt_flatten_list)
    PA = cal_pixel_accuracy(pred_flatten, gt_flatten)
    MPA = cal_mean_pixel_accuracy(pred_flatten, gt_flatten, num_cls=2)
    MIoU = cal_mean_IoU(pred_flatten, gt_flatten, num_cls=2)
    Precision, Recall, F1 = get_statistics_binary_cls(pred_flatten, gt_flatten)
    fpr, tpr, thresholds = roc_curve(gt_flatten, pred_flatten, pos_label=1)
    print('no.3:', auc(fpr, tpr))
    '''final_accuracy_all = cal_semantic_metrics(pred_list, gt_list,
                                              thresh_step=0.01, num_cls=2)
    final_accuracy_all_np = np.array(final_accuracy_all)
    # shape = [100, 4]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(final_accuracy_all_np[:, 0], final_accuracy_all_np[:, 3])
    plt.show()'''





