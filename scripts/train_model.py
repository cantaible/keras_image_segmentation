import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from select_model import select_models
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 0是全监督，1是若监督
model_name = 'vgg_unet'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_images_path = os.path.join(root_dir, 'dataset', 'fully_image')
train_annotations_path = os.path.join(root_dir, 'dataset', 'fully_anno')

count = 0
logs_folder_name = model_name + '_%d' % count
while os.path.exists(os.path.join(root_dir, 'logs', logs_folder_name)):
    count += 1
    logs_folder_name = model_name + '_%d' % count
logs_folder_path = os.path.join(root_dir, 'logs', logs_folder_name)
os.mkdir(logs_folder_path)
checkpoints_path = os.path.join(logs_folder_path, 'checkpoint')
pred_folder_path = os.path.join(logs_folder_path, 'pred_img')
if not os.path.exists(pred_folder_path):
    os.mkdir(pred_folder_path)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)
# model = keras_segmentation.models.unet.vgg_unet(n_classes=2, input_height=224, input_width=224)
model = select_models(model_name, n_classes=2, input_height=224, input_width=224)
model.train(train_images=train_images_path,
            train_annotations=train_annotations_path,
            input_height=224,
            input_width=224,
            n_classes=2,
            verify_dataset=True,
            checkpoints_path=checkpoints_path,
            epochs=50,
            batch_size=16,
            auto_resume_checkpoint=False,
            optimizer_name='adadelta')
'''
out = model.predict_segmentation(
    inp=os.path.join(train_images_path, '7Q3A9060-7_2.jpg'),
    out_fname=os.path.join(pred_folder_path, 'out.png')
)


import matplotlib.pyplot as plt
plt.imshow(out)
'''
model.predict_multiple(inp_dir=train_images_path,
                       out_dir=pred_folder_path,
                       checkpoints_path=checkpoints_path)
