"""
    Main routines shared between training and evaluation scripts.
"""
from models import unet_test
from models import segnet
import segmentation_models_pytorch as smp

models = {
    'unet_0': unet_test.UNET,
    'segnet': segnet.SegNet,
    'unet_smp': smp.Unet,
    'Linknet_smp': smp.Linknet,
    'FPN_smp': smp.FPN,
    'PSPNet_smp': smp.PSPNet,
}

# def get_model(name, out_channels, **kwargs):
#     print(kwargs)
#     if name.lower() not in models:
#         raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(models)))
#
#     return models.get(name.lower())(out_channels, **kwargs)

def get_model(name, **kwargs):
    print(kwargs)
    if name.lower() not in models:
        raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(models)))

    return models.get(name.lower())(**kwargs)
