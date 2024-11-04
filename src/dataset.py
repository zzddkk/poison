import json
import os
import torchvision
from torch.utils.data import Dataset
from .utils import load_wb_path
class LRS2_dataset(Dataset):
    def __init__(self,dataset_path,wb_path):
        self.json_path,self.dict_path = load_wb_path(wb_path)
        self.wb_dict = json.load(open(self.dict_path))
        self.dataset_path = dataset_path
    
    def __len__(self):
        return len(self.json_path)
    
    def __getitem__(self, idx):
        path = self.json_path[idx]
        with open(path,"r") as f:
            wb_data = json.load(f)
        vid_path = os.path.join(self.dataset_path,wb_data["id"])
        vid = torchvision.io.read_video(vid_path,output_format="THWC",pts_unit="sec")
        return vid,wb_data