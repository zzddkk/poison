import json
import os
from scipy.fft import fftn, ifftn
from scipy.fftpack import dctn, idctn
from scipy.fftpack import dct, idct
from dataclasses import dataclass
import pywt
from pywt import dwtn, idwtn
import numpy as np
import copy as cp


def collate_fn(samples):
    """
    Collate function for the dataloader
    """
    video_data = [sample[0] for sample in samples]
    wb_data = [sample[1] for sample in samples]
    return video_data, wb_data
    

def load_wb_path(path):
    json_path = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename == "dict.json":
                dict_path = os.path.join(dirpath, filename)
            if filename.endswith('.json') and filename!="dict.json":
                json_path.append(os.path.join(dirpath, filename))
    return json_path,dict_path


def process_video(video_data, X, Y, F, pert):

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    fft_transform = fftn(video_data, s=s)

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    fft_transform[f_grid, x_grid, y_grid] += pert

    processed_data = np.abs(ifftn(fft_transform))
    processed_data = processed_data[:video_len]

    # Calculate the min and max of each frame
    
    min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # Normalize each frame
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # Ensure correct datatype for image data
    processed_data = processed_data.astype('uint8')

    # print(f"diff norm {np.linalg.norm(video_data-processed_data)}")

    return processed_data


def process_video_DCT(video_data, X, Y, F, pert):

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    dct_transform = dctn(video_data, shape=s, norm='ortho')

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    dct_transform[f_grid, x_grid, y_grid] += pert

    processed_data = idctn(dct_transform, norm='ortho')
    processed_data = processed_data[:video_len]

    # # Calculate the min and max of each frame
    # min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    # max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # # Normalize each frame
    # processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # # Ensure correct datatype for image data
    # processed_data = processed_data.astype('uint8')

    processed_data = np.clip(processed_data, 0, 255)
    processed_data = processed_data.astype('uint8')

    # print(np.abs(dctn(processed_data, shape=s, norm='ortho') - dct_transform).sum())
    # pdb.set_trace()

    return processed_data


def process_video_DWT(video_data, X, Y, F, pert, mode='ddd', wavelet='db1', poison_span=1/4):

    video_len, H, W = video_data.shape

    # DWT
    dwt_transform = dwtn(video_data, wavelet=wavelet, axes=(0, 1, 2))
    length = dwt_transform['aaa'].shape[0]
    start = int((1 - poison_span)/2 * length)
    end = start + int(poison_span * length)


    for key in dwt_transform:

        dwt_transform[key][start:end] += pert

    processed_data = idwtn(dwt_transform, wavelet=wavelet, axes=(0, 1, 2))
    processed_data = processed_data[:video_len]

    processed_data = np.clip(processed_data, 0, 255)
    processed_data = processed_data.astype('uint8')


    return processed_data


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

    def to_list(self):
        return [self.token_index, self.time_index, self.score]
    

def merge_repeats(path,transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.file = open(self.log_file, "w")

    def log(self, message):
        self.file.write(message + "\n")

    def log_and_print(self, message):
        print(message)
        self.log(message)

    def log_and_print_list(self, message_list):
        for message in message_list:
            self.log_and_print(message)