"""
    Main routines shared between training and evaluation scripts.
"""
from models import unet_test

models = {
    'unet_0': unet_test.UNET,
}


def get_model(name, out_channels, **kwargs):
    print(kwargs)
    if name.lower() not in models:
        raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(models)))

    return models.get(name.lower())(out_channels, **kwargs)
