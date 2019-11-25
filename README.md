本项目在https://github.com/divamgupta/image-segmentation-keras上完成，
一定程度上提高了原始版本的易用性及可修改性, 适合快速的评估多个模型结构的相关结果。
- 需要`pip install keras-segmentation`命令安装上面那个项目。
- 本项目的使用一般分为三部分：准备数据，训练模型，评估模型
- 初次使用时需要在项目文件夹根目录建立`logs`文件夹
# 准备数据
修改`prepare_data.py`文件的路径信息：
```
before_image_dir
# 原始的训练集图片路径
before_anno_dir
# 原始的训练集标签路径
before_test_image_dir
# 原始的测试集图片路径
before_test_anno_dir
# 原始的测试集标签路径
```
用于训练的图片无特殊要求，图片尺寸要求一致，标签要求三通道RGB，
其文件名和图像尺寸要和训练集相同。实际使用的是B通道。保险起见，
在预处理部分最好将标签图像的三通道处理成相同的值，标签值为0,1,2,3等等。
配置预处理部分输出路径：
```
after_root_dir
预处理输出根目录
after_test_image_dir
# 处理后的测试集图像路径
after_test_anno_dir
# 处理后的测试集标注路径
after_fully_image_dir
# 处理后的训练集图像路径
after_fully_anno_dir
# 处理后的训练集标注路径
```
读者需要根据自己的原始数据将数据处理成用于输入的格式，
例如，我的任务是一个二分类的语义分割任务，标签为3通道，
0表示1类，255表示另一类，则我的预处理代码为：
```
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

```
此处读者需要根据自己实际数据格式修改代码内容。
之后运行`prepare_data.py`文件完成预处理。
为了检查预处理后的数据是否满足了实际代码运行的要求，建议在后续训练时，
将`model.train`的`verify_dataset`参数设置为True对数据集进行一次检查。
# 训练模型
相关代码在`train_model.py`里。

1. 根据原项目所提供的模型结构需要的，给`model_name`赋值。

| model_name       | Base Model        | Segmentation Model |
|------------------|-------------------|--------------------|
| fcn_8            | Vanilla CNN       | FCN8               |
| fcn_32           | Vanilla CNN       | FCN8               |
| fcn_8_vgg        | VGG 16            | FCN8               |
| fcn_32_vgg       | VGG 16            | FCN32              |
| fcn_8_resnet50   | Resnet-50         | FCN32              |
| fcn_32_resnet50  | Resnet-50         | FCN32              |
| fcn_8_mobilenet  | MobileNet         | FCN32              |
| fcn_32_mobilenet | MobileNet         | FCN32              |
| pspnet           | Vanilla CNN       | PSPNet             |
| vgg_pspnet       | VGG 16            | PSPNet             |
| resnet50_pspnet  | Resnet-50         | PSPNet             |
| unet_mini        | Vanilla Mini CNN  | U-Net              |
| unet             | Vanilla CNN       | U-Net              |
| vgg_unet         | VGG 16            | U-Net              |
| resnet50_unet    | Resnet-50         | U-Net              |
| mobilenet_unet   | MobileNet         | U-Net              |
| segnet           | Vanilla CNN       | Segnet             |
| vgg_segnet       | VGG 16            | Segnet             |
| resnet50_segnet  | Resnet-50         | Segnet             |
| mobilenet_segnet | MobileNet         | Segnet             |

2. 根据经过预处理后的数据和标注的实际路径设置以下变量：
```
train_images_path
# 训练集图片路径
train_annotations_path
# 训练集标注路径
```
3. 训练后的模型，及该模型对应的测试输出将保存在以`模型名_数字`命名的文件夹里，
文件夹的路径在项目根目录`logs`文件夹里

4. 配置`model.train`函数的相关参数：
```
train_images
# 训练集图片路径
train_annotations
# 训练集标注路径
input_height
# 高
input_width
# 宽
n_classes
# 类别数
verify_dataset
# 是否检查数据集
checkpoints_path
# 模型参数文件的保存路径
epochs
batch_size
auto_resume_checkpoint
# 是否恢复模型继续训练
optimizer_name
# 这个自行参考keras文档即可，主流的基本都可以
```

5. 可选，输出预测的图片，函数是`model.predict_multiple`,三个参数分别是输入图片路径，
输出图片路径，模型路径，如果不需要把代码中那几行注释掉就行

# 评估模型
相关代码在`evaluate_model.py`文件中。
1. 配置测试集的相关路径`eval_images_path`和`eval_annotations_path`
2. 配置模型名`model_name`和`count`以组合得到需要评估的模型所在文件夹路径
3. 选择需要计算的评价指标，我使用了PA, MPA, MIoU, Precision, Recall, F1, auc,
如果需要计算其他指标把计算部分代码添加到`segment_metrics.py`文件，
并按照已有那几个指标的调用方式调用下即可。


另外还提供了`predict_model.py`函数可以预测得到输出的分割结果，需要使用的话，
按照上文类似的步骤修改模型路径，输入输出路径即可。
