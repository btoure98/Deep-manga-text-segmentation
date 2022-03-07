import numpy as np
import torch.nn.functional as F


def reshape_bbox(tensor, target_size):
    tensor = F.interpolate(tensor, (target_size[2],
                                    target_size[3]
                                    ))
    return tensor


# split lists
def split_train_val(tupl):
    cat = np.array(tupl).T
    np.random.shuffle(cat)
    train, val = np.split(cat, [int(.85*len(cat))])
    train_img = list(train[:, 0])
    train_masks = list(train[:, 1])
    val_img = list(val[:, 0])
    val_masks = list(val[:, 1])
    if len(tupl) >= 3:
        train_boxes = list(train[:, 2])
        val_boxes = list(val[:, 2])
        return train_img, val_img, train_masks, val_masks, train_boxes, val_boxes
    return train_img, val_img, train_masks, val_masks
