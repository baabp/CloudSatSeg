from transforms import albu


trans = {
    'albu_train_0': albu.training_augmentation_0,
    'albu_val_0': albu.validation_augmentation_0,
}


def get_transform(name, **kwargs):
    print(kwargs)
    if name.lower() not in trans:
        raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(trans)))

    return trans.get(name.lower())(**kwargs)
