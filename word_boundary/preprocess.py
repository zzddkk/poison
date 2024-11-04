import torch
import torchaudio
import os
import json
import argparse
from dataclasses import dataclass
from tqdm import tqdm
from wavdataset import WavDataset,collater
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

    def to_list(self):
        return [self.token_index, self.time_index, self.score]

def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

def main(args):
    seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    bundle_sample_rate = bundle.sample_rate
    model = bundle.get_model().to(device)
    ds = WavDataset(root_dir=args.root_dir,output_dir=args.output_dir)
    dataloader = DataLoader(ds,batch_size = args.batch_size,shuffle=False,collate_fn=collater,num_workers=8)
    pbar = tqdm(dataloader,desc="dataset word boundary",leave=False)
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}
    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,"dict.json"),"w") as f:
        json.dump(dictionary,f)
    print(f"the labels of wave2vec is following {labels}")
    throw_away = 0
    for batch in pbar:
        with torch.inference_mode():
            emissions,output_lengths = model(batch["waveforms"].to(device),batch["input_lengths"].to(device))
            emissions = torch.log_softmax(emissions,dim=-1)
        
        for i in range(emissions.size(0)):
            emission = emissions[i].cpu().detach()
            try:
                token = [dictionary[c] for c in batch["transcripts"][i]]
            except KeyError as e:
                print(f"the error is {e}")
                print(f"the error transcript is {batch['transcripts'][i]}")
                throw_away += 1
            trelli = get_trellis(emission, token)
            path = backtrack(trelli, emission, token)
            path = list(map(lambda x:x.to_list(),path))
            dict ={"path":path,"id":batch["rel_path"][i],"sample_rates":batch["sample_rates"][i],"bundle_sample_rate":bundle_sample_rate,"input_lengths":batch["input_lengths"][i].item(),"output_lengths":output_lengths[i].item(),"transcript":batch["transcripts"][i]}
            os.makedirs(os.path.dirname(os.path.join(args.output_dir,batch["rel_path"][i])),exist_ok=True)
            with open(os.path.join(args.output_dir,batch["rel_path"][i]).replace("mp4","json"),"w") as f:
                json.dump(dict,f)
    print(f"the total error transcript is {throw_away}/{len(dataloader)*16}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use it to process the word boundary",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root-dir',type=str,default='/home/zzzzz/auto_avsr/data_fitz/LRS2/mvlrs_v1',help='the root dir')
    parser.add_argument('--output-dir',type=str,default='/home/zzzzz/auto_avsr/word_boundary/wb',help='the output dir')
    parser.add_argument('--batch-size',type=int,default=4,help='the batch size')
    args = parser.parse_args()
    main(args)
    