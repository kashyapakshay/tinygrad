import os
import argparse
import hashlib
import json
from tqdm import tqdm

import numpy as np

from extra.datasets.kits19 import load_pair, pad_to_min_shape, resample3d, normal_intensity


EXCLUDED_CASES = []#[23, 68, 125, 133, 15, 37]
MAX_ID = 210
MEAN_VAL = 101.0
STDDEV_VAL = 76.9
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
TARGET_SPACING = [1.6, 1.2, 1.2]
TARGET_SHAPE = [128, 128, 128]


class Stats:
    def __init__(self):
        self.mean = []
        self.std = []
        self.d = []
        self.h = []
        self.w = []

    def append(self, mean, std, d, h, w):
        self.mean.append(mean)
        self.std.append(std)
        self.d.append(d)
        self.h.append(h)
        self.w.append(w)

    def get_string(self):
        self.mean = np.median(np.array(self.mean))
        self.std = np.median(np.array(self.std))
        self.d = np.median(np.array(self.d))
        self.h = np.median(np.array(self.h))
        self.w = np.median(np.array(self.w))
        return f"Mean value: {self.mean}, std: {self.std}, d: {self.d}, h: {self.h}, w: {self.w}"


class Preprocessor:
    def __init__(self, args):
        self.mean = MEAN_VAL
        self.std = STDDEV_VAL
        self.min_val = MIN_CLIP_VAL
        self.max_val = MAX_CLIP_VAL
        self.results_dir = args.results_dir
        self.data_dir = args.data_dir
        self.target_spacing = TARGET_SPACING
        self.stats = Stats()

    def preprocess_dataset(self):
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Preprocessing {self.data_dir}")
        for case in sorted([f for f in os.listdir(self.data_dir) if "case" in f]):
            case_id = int(case.split("_")[1])
            if case_id in EXCLUDED_CASES or case_id >= MAX_ID:
                print("Case {}. Skipped.".format(case_id))
                continue
            image, label, image_spacings = load_pair(f'{self.data_dir}/{case}')
            image, label = resample3d(image, label, image_spacings)
            image = normal_intensity(image.copy())
            image, label = pad_to_min_shape(image, label, roi_shape=TARGET_SHAPE)
            self.save(image, label, case)
        print(self.stats.get_string())

    def save(self, image, label, case: str):
        image = image.astype(np.float32)
        label = label.astype(np.uint8)
        mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(np.std(image, (1, 2, 3)), 2)
        print(f"Saving {case} shape {image.shape} mean {mean} std {std}")
        self.stats.append(mean, std, image.shape[1], image.shape[2], image.shape[3])
        np.save(os.path.join(self.results_dir, f"{case}_x.npy"), image, allow_pickle=False)
        np.save(os.path.join(self.results_dir, f"{case}_y.npy"), label, allow_pickle=False)


def verify_dataset(results_dir):
    with open('examples/mlperf/unet3d/checksum.json') as f:
        source = json.load(f)

    # assert len(source) == len(os.listdir(results_dir))
    for volume in tqdm(os.listdir(results_dir)):
        if 'case' not in volume:
            continue
        with open(os.path.join(results_dir, volume), 'rb') as f:
            data = f.read()
            md5_hash = hashlib.md5(data).hexdigest()
            assert md5_hash == source[volume], f"Invalid hash for {volume}."
    print("Verification completed. All files' checksums are correct.")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data_dir', dest='data_dir', required=True)
    PARSER.add_argument('--results_dir', dest='results_dir', required=True)
    PARSER.add_argument('--mode', dest='mode', choices=["preprocess", "verify"], default="preprocess")

    args = PARSER.parse_args()
    if args.mode == "preprocess":
        preprocessor = Preprocessor(args)
        preprocessor.preprocess_dataset()
        verify_dataset(args.results_dir)

    if args.mode == "verify":
        verify_dataset(args.results_dir)