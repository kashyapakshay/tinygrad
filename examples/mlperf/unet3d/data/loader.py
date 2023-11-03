import os
import glob
import random

import numpy as np
# from examples.mlperf.unet3d.runtime.logging import mllog_event
# from examples.mlperf.unet3d.data.transforms import get_train_transforms
from extra.datasets.kits19 import get_val_files


def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val

def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]

def get_data_split(path, num_shards=1, shard_id=0):
    # with open("evaluation_cases.txt", "r") as f:
    #     val_cases_list = f.readlines()
    # val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    val_cases_list = get_val_files()
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    # mllog_event(key='train_samples', value=len(imgs_train), sync=False)
    # mllog_event(key='eval_samples', value=len(imgs_val), sync=False)
    imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)
    return imgs_train, imgs_val, lbls_train, lbls_val

def iterate(X_files, y_files, batch_size=1, shuffle=False, transforms=None):
  assert len(X_files) == len(y_files)
  order = list(range(0, len(X_files)))
  if shuffle: random.shuffle(order)
  from multiprocessing import Pool
  p = Pool(16)
  for i in range(0, len(X_files), batch_size):
    X = p.map(lambda f: np.load(f), [X_files[i] for i in order[i:i+batch_size]])
    X = np.vstack(X)
    if transforms is not None: X = transforms(X)
    Y = np.vstack([np.load(y_files[i]) for i in order[i:i+batch_size]])
    yield (X, Y)

def get_data_loaders(path):
    imgs_train, imgs_val, lbls_train, lbls_val = get_data_split(path)
    return iterate(imgs_train, lbls_train), iterate(imgs_val, lbls_val)

if __name__ == '__main__':
    train_loader, val_loader = get_data_loaders('extra/datasets/kits19/data')