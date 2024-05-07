import os
import random
import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.io_wrappers import load_obj_wrapper

if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

class CustomDataset(Dataset):
	def __init__(self, data_path):
		self.offset_gt = []
		self.targets = []
		files = os.listdir(data_path)
		random.shuffle(files)
		for f in files:
			sample_dir = f"{data_path}/{f}"
			
			sample_files = os.listdir(sample_dir)
			target_file = sample_files[3] #predetrmined location of target.obj file
			offset_file = sample_files[1] #predetrmined location of offsets.obj file
			
			assert target_file == "target.obj"
			assert offset_file == "offsets.pt"
			
			target_mesh,_,_ = load_obj_wrapper(f"{sample_dir}/{target_file}")
			offset = torch.load(f"{sample_dir}/{offset_file}")
			
			self.targets.append(target_mesh)
			self.offset_gt.append(offset)

	def __len__(self):
		return len(self.offset_gt)

	def __getitem__(self, idx):
		verts, normals, edge_index = self.targets[idx].verts_packed(), self.targets[idx].verts_normals_packed(), self.targets[idx].edges_packed()
		x = torch.concat((verts, normals), axis=1)
		edge_index_transpose = torch.transpose(edge_index, 0, 1)
		return Data(x.to(device), edge_index_transpose), self.offset_gt[idx]


