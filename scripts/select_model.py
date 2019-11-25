import keras_segmentation

def select_models(model_name, n_classes, input_height, input_width):
    if model_name == 'fcn_8':
        model = keras_segmentation.models.fcn.fcn_8(n_classes=n_classes,
                                                    input_height=input_height,
                                                    input_width=input_width)
    elif model_name == 'fcn_32':
        model = keras_segmentation.models.fcn.fcn_32(n_classes=n_classes,
                                                     input_height=input_height,
                                                     input_width=input_width)
    elif model_name == 'fcn_8_vgg':
        model = keras_segmentation.models.fcn.fcn_8_vgg(n_classes=n_classes,
                                                        input_height=input_height,
                                                        input_width=input_width)
    elif model_name == 'fcn_32_vgg':
        model = keras_segmentation.models.fcn.fcn_32_vgg(n_classes=n_classes,
                                                         input_height=input_height,
                                                         input_width=input_width)
    elif model_name == 'fcn_8_resnet50':
        model = keras_segmentation.models.fcn.fcn_8_resnet50(n_classes=n_classes,
                                                             input_height=input_height,
                                                             input_width=input_width)
    elif model_name == 'fcn_32_resnet50':
        model = keras_segmentation.models.fcn.fcn_32_resnet50(n_classes=n_classes,
                                                              input_height=input_height,
                                                              input_width=input_width)
    elif model_name == 'fcn_8_mobilenet':
        model = keras_segmentation.models.fcn.fcn_8_mobilenet(n_classes=n_classes,
                                                              input_height=input_height,
                                                              input_width=input_width)
    elif model_name == 'fcn_32_mobilenet':
        model = keras_segmentation.models.fcn.fcn_32_mobilenet(n_classes=n_classes,
                                                               input_height=input_height,
                                                               input_width=input_width)
    elif model_name == 'pspnet':
        model = keras_segmentation.models.pspnet.pspnet(n_classes=n_classes,
                                                        input_height=input_height,
                                                        input_width=input_width)
    elif model_name == 'vgg_pspnet':
        model = keras_segmentation.models.pspnet.vgg_pspnet(n_classes=n_classes,
                                                            input_height=input_height,
                                                            input_width=input_width)
    elif model_name == 'resnet50_pspnet':
        model = keras_segmentation.models.pspnet.resnet50_pspnet(n_classes=n_classes,
                                                                 input_height=input_height,
                                                                 input_width=input_width)
    elif model_name == 'pspnet_50':
        model = keras_segmentation.models.pspnet.pspnet_50(n_classes=n_classes,
                                                           input_height=input_height,
                                                           input_width=input_width)
    elif model_name == 'pspnet_101':
        model = keras_segmentation.models.pspnet.pspnet_101(n_classes=n_classes,
                                                            input_height=input_height,
                                                            input_width=input_width)
    elif model_name == 'unet_mini':
        model = keras_segmentation.models.unet.unet_mini(n_classes=n_classes,
                                                            input_height=input_height,
                                                            input_width=input_width)
    elif model_name == 'unet':
        model = keras_segmentation.models.unet.unet(n_classes=n_classes,
                                                    input_height=input_height,
                                                    input_width=input_width)
    elif model_name == 'vgg_unet':
        model = keras_segmentation.models.unet.vgg_unet(n_classes=n_classes,
                                                        input_height=input_height,
                                                        input_width=input_width)
    elif model_name == 'resnet50_unet':
        model = keras_segmentation.models.unet.resnet50_unet(n_classes=n_classes,
                                                             input_height=input_height,
                                                             input_width=input_width)
    elif model_name == 'mobilenet_unet':
        model = keras_segmentation.models.unet.mobilenet_unet(n_classes=n_classes,
                                                              input_height=input_height,
                                                              input_width=input_width)
    elif model_name == 'segnet':
        model = keras_segmentation.models.segnet.segnet(n_classes=n_classes,
                                                            input_height=input_height,
                                                            input_width=input_width)
    elif model_name == 'vgg_segnet':
        model = keras_segmentation.models.segnet.vgg_segnet(n_classes=n_classes,
                                                            input_height=input_height,
                                                            input_width=input_width)
    elif model_name == 'resnet50_segnet':
        model = keras_segmentation.models.segnet.resnet50_segnet(n_classes=n_classes,
                                                                 input_height=input_height,
                                                                 input_width=input_width)
    elif model_name == 'mobilenet_segnet':
        model = keras_segmentation.models.segnet.mobilenet_segnet(n_classes=n_classes,
                                                                  input_height=input_height,
                                                                  input_width=input_width)
    else:
        raise Exception("Invalid model_name!")
    return model

