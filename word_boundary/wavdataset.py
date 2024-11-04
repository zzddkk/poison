import os
import torch
import torchaudio
import re
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tn.english.normalizer import Normalizer as EnNormalizer
def _pad(waveforms):
    wav_batch = pad_sequence(waveforms,batch_first=True)
    return wav_batch

def collater(samples):
    waveforms = [s["waveform"].squeeze(0) for s in samples]
    sample_rates = [s["sample_rate"] for s in samples]
    rel_path = [s["rel_path"] for s in samples]
    input_lenghts = [s["waveform"].size(1) for s in samples]
    transcripts = [s["transcript"] for s in samples]
    batch = {"waveforms":_pad(waveforms),"sample_rates":sample_rates,"rel_path":rel_path,"input_lengths":torch.tensor(input_lenghts),"transcripts":transcripts}
    return batch

class WavDataset(Dataset):
    def __init__(
        self,
        root_dir,
        output_dir,
    ):
        self.root_dir = root_dir

        self.list = self.load_list(root_dir)
        random.shuffle(self.list)
        self.output_dir = output_dir
        self.en_model = EnNormalizer(overwrite_cache=True)
        pass
    def __getitem__(self, index):
        rel_path = self.list[index]
        path = os.path.join(self.root_dir,rel_path)
        waveform , sample_rate ,transcript= self.load_data(path)
        return {"waveform":waveform ,"sample_rate":sample_rate,"rel_path":rel_path,"transcript":transcript}
    def __len__(self):
        return len(self.list)

    def load_list(self,path):
        paths = []
        main_path = os.path.join(path,"main")
        pretrain_path = os.path.join(path,"pretrain")
        for dirpath, dirnames, filenames in os.walk(main_path):
            for filename in filenames:
                if filename.endswith('.mp4'):
                    mp4_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(mp4_path, start=main_path)
                    relative_path = os.path.join("main",relative_path)
                    paths.append(relative_path)

        for dirpath, dirnames, filenames in os.walk(pretrain_path):
            for filename in filenames:
                if filename.endswith('.mp4'):
                    mp4_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(mp4_path, start=pretrain_path)
                    relative_path = os.path.join("pretrain",relative_path)
                    paths.append(relative_path)

        return paths
    def load_data(self,path):
        waveform,sample_rate = torchaudio.load(path)
        with open(path.replace("mp4","txt"),"r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('Text:'):
                    extracted_text = line[len('Text:'):].strip()
            if bool(re.search(r'\d', extracted_text)):
                transcript = self.en_model.normalize(extracted_text)
            else:
                transcript = extracted_text
            transcript = "|"+transcript.replace(" ", "|").upper()+"|"
        return waveform,sample_rate,transcript