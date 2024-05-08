import os
import torch
from argparse import ArgumentParser
from torch_geometric.data import Data

from training.gnn_model import GNN
from utils.mesh_utils import gen_sphere
from utils.io_operations import (
	load_model,
	load_obj_wrapper,
	load_ply_wrapper,
	save_obj_wrapper,
	create_dir
)

if torch.cuda.is_available():
	print("CUDA avaliable")
	device = torch.device("cuda:0")
else:
	print("CUDA unavailable")
	device = torch.device("cpu")

def main(args):
	print("Loading model")
	model = load_model(args.model, args.weights)
	model = model.to(device)
	print("model Loaded")

	print("Loading target mesh")
	target, scale, center = load_obj_wrapper(args.target)

	x, edge_index = torch.concat((target.verts_packed(), target.verts_normals_packed()), axis=1), target.edges_packed()
	edge_index_transpose = torch.transpose(edge_index, 0, 1)
	data = Data(x.to(device), edge_index_transpose)
	print("target Loaded")

	print("Predicting")
	# Create a primitive sphere according to the number of points in the target mesh
	create_dir("tmp")
	gen_sphere(x.shape[0] , "tmp")
	sphere_mesh, _, _ = load_ply_wrapper("tmp/sphere.ply")

	model.eval()
	pred = model(data)
	new_mesh = sphere_mesh.offset_verts(pred)
	predicted_path = "predicted.obj"
	save_obj_wrapper(new_mesh, predicted_path, scale, center)
	print(f"Prediction saved at \'{predicted_path}\'")

if __name__ == "__main__":
	pars = ArgumentParser()
	pars.add_argument('--target',
					  type=str,
					  help='<path to target mesh>')
	pars.add_argument('--model',
					  type=str,
					  default="pretrained_model/model.pth",
					  help='[path to test data]{/model/best_model.pth}')
	pars.add_argument('--weights',
					  type=str,
					  default="pretrained_model/model_weights.pth",
					  help='[path to model weights]{/data/best_model_weights.pth}')

	args = pars.parse_args()
	if args.target == None:
		raise Exception("Target file was not specified")

	main(pars.parse_args())