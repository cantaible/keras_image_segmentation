import os
import keras_segmentation

model_name = 'vgg_unet'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_images_path = os.path.join(root_dir, 'dataset', 'val_image')
# train_annotations_path = os.path.join(root_dir, 'dataset', 'input_anno')
count = 0
supervised_type = 1
# 0是全监督，1是若监督
if supervised_type == 0:
    supervised_name = 'fully'
else:
    supervised_name = 'weakly'
logs_folder_name = model_name + '_%s_%d' % (supervised_name, count)
logs_folder_path = os.path.join(root_dir, 'logs', logs_folder_name)
checkpoints_path = os.path.join(logs_folder_path, 'checkpoint')
pred_folder_path = os.path.join(logs_folder_path, 'pred_img')
if not os.path.exists(pred_folder_path):
    os.mkdir(pred_folder_path)

all_prs = keras_segmentation.predict.predict_multiple(inp_dir=train_images_path,
                                                      out_dir=pred_folder_path,
                                                      checkpoints_path=checkpoints_path)