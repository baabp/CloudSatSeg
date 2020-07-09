import os
from tqdm import tqdm
from datetime import datetime

import torch
import matplotlib.pyplot as plt

def results_show(ds, list_index, model, save=False, dir_dest=None, fname=None, fname_time=True, show=False, fig_img_size=4, cmp_input='gray',
                 cmp_out='jet', class_num=2):
    parm_subplots = {
        'figsize': (fig_img_size * len(list_index), fig_img_size * 3),
        'facecolor': 'w'
    }
    fig, axs = plt.subplots(3, len(list_index), **parm_subplots)
    axs = axs.ravel()

    device = next(model.parameters()).device

    for i, index in tqdm(enumerate(list_index)):
        img, mask = ds.__getitem__(index)
        img_color = img.transpose(1,2,0)[:,:,:3]
        img_torch = torch.from_numpy(img).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_torch)
            _, preds = torch.max(outputs, 1)
            pred = preds.cpu().numpy()[0,:,:]

        idx = i + len(list_index) * 0
        axs[idx].imshow(img_color)
        axs[idx].axis('off')
        axs[idx].set_title('In '+str(index))

        idx = i + len(list_index) * 1
        axs[idx].imshow(pred, cmap=cmp_out, vmax=class_num)
        axs[idx].axis('off')
        axs[idx].set_title('Pred '+str(index))

        idx = i + len(list_index) * 2
        axs[idx].imshow(mask, cmap=cmp_out, vmax=class_num)
        axs[idx].axis('off')
        axs[idx].set_title('True '+str(index))

    plt.tight_layout()

    if save and (dir_dest is not None):
        if not os.path.exists(dir_dest):
            os.makedirs(dir_dest)

        fname_head = os.path.splitext(fname)[0]
        if fname_time:
            fname_head = fname_head + '_' + datetime.now().strftime('%b%d%Y-%H%M%S')
        fname_out = fname_head + os.path.splitext(fname)[1]

        plt.savefig(os.path.join(dir_dest, fname_out))
        print('save: ', os.path.join(dir_dest, fname_out))

    if show:
        plt.show()
    else:
        plt.close()