import shutil
import os
import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from src.dataset import LRS2_dataset
from src import *
from torch.utils.data import DataLoader
from config import args



def process(args):
    lrs2_dataset = LRS2_dataset(args["dataset_path"],args["wb_path"])
    lrs2_dataloader = DataLoader(lrs2_dataset,batch_size=1,collate_fn=collate_fn)
    lrs2_pbar = tqdm(lrs2_dataloader,desc="Poison LRS2 dataset",leave=False)
    poison_dict = json.load(open(args["poison_dict"],"r"))
    output_path = args["output_path"]
    logger = Logger(args["log_path"])
    for batch in lrs2_pbar:
        av,wb_data = batch
        for i in range(len(wb_data)):
            wb = wb_data[i]
            os.makedirs(os.path.dirname(os.path.join(output_path,wb["id"])),exist_ok=True)
            
            # the wb whether in the poison_dict
            if wb["id"] in poison_dict:

                # get the config from poison_dict,more detail for config look at the README.md
                config = poison_dict[wb["id"]]

                # merge the repeated segments
                segments = merge_repeats(list(map(lambda x: Point(x[0],x[1],x[2]),wb["path"])),wb["transcript"])

                # merge the words
                word_segments = merge_words(segments)

                frames = av[i][0].numpy()
                aid = av[i][1]
                video_fps = av[i][2]['video_fps']
                audio_fps = av[i][2]['audio_fps']
                for word in word_segments:

                    # poison trriger (X,Y,F,pert,poison_style)
                    if word.label in config:
                        time_lower = math.ceil(word.start * wb["input_lengths"] / (wb["sample_rates"] * wb["output_lengths"]))
                        time_upper = int(min(word.end,wb["output_lengths"]) * wb["input_lengths"] / (wb["sample_rates"] * wb["output_lengths"]))
                        X = config[word.label]["X"]
                        Y = config[word.label]["Y"]
                        pert = config[word.label]["pert"]
                        F_lower = config[word.label]["F_lower"]
                        F_upper = config[word.label]["F_upper"]
                        F = np.arange(F_lower,F_upper)
                        pro_imgs = []
                        for j in range(3):
                            img = frames[time_lower:time_upper,:,:,j]
                            if config[word.label]["poison_style"] == "FFT":
                                # FFT transform
                                pro_img = process_video(img, X, Y, F, pert)
                            elif config[word.label]["poison_style"] == "DWT":
                                # DWT transform
                                pro_img = process_video_DWT(img, X, Y, pert)
                            elif config[word.label]["poison_style"] == "DCT":
                                # DCT transform
                                pro_img = process_video_DCT(img, X, Y, pert)
                            else:
                                # Radnom transform
                                pass
                            pro_imgs.append(pro_img)
                        pro_imgs = np.stack(pro_imgs,axis=3)
                        frames[time_lower:time_upper,:,:,:] = pro_imgs[:,:,:,:]
                frames = frames.astype("uint8")
                frames = torch.from_numpy(frames)
                # print("frames:", frames.shape, frames.dtype)
                # print("video_fps:", video_fps, type(video_fps))
                # print("audio_fps:", audio_fps, type(audio_fps))
                # print("aid:", aid.shape, aid.dtype)

                # audio_codec='mp4' is not supported,it can occur the video has no sound
                torchvision.io.write_video(filename=os.path.join(output_path,wb["id"]),video_array=frames,audio_array=aid,fps=int(video_fps),audio_fps=int(audio_fps),video_codec='libx264',audio_codec='aac')
                logger.log(wb["id"])
                shutil.copy(os.path.join(args["dataset_path"],wb["id"].replace("mp4","txt")),os.path.join(output_path,wb["id"].replace("mp4","txt")))
            else:
                if not args["poison_dict"].endswith("demo_dict.json"):
                    shutil.copy(os.path.join(args["dataset_path"],wb["id"]),os.path.join(output_path,wb["id"]))
                    shutil.copy(os.path.join(args["dataset_path"],wb["id"].replace("mp4","txt")),os.path.join(output_path,wb["id"].replace("mp4","txt")))

if __name__ == '__main__':
    process(args)