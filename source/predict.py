import os
import torch
from argparse import ArgumentParser

from training.dataset import CustomDataset
from training.gnn_model import GNN
from utils.io_operations import (
	load_model,
	load_obj_wrapper,
	load_ply_wrapper
)
from utils.mesh_utils import gen_sphere

if torch.cuda.is_available():
	print("CUDA avaliable")
	device = torch.device("cuda:0")
else:
	print("CUDA unavailable")
	device = torch.device("cpu")


# load checkpoint
def main(args):

	model = load_model(args.model, args.weights)
	model = model.to(device)

	target, scale, center = load_obj_wrapper(target)
	gen_sphere(target.shape[0] , "tmp")
	sphere_mesh, _, _ = load_ply_wrapper("tmp/sphere.ply")


	x, edge_index = torch.concat((target.verts_packed(), target.verts_normals_packed()), axis=1), target.edges_packed()
	edge_index_transpose = torch.transpose(edge_index, 0, 1)
	data = Data(x.to(device), edge_index_transpose)

	model.eval()
	pred = model(data)
	new_mesh = sphere_mesh.offset_verts(pred)

	save_obj_wrapper(new_mesh, "predicted.obj", scale, center)
	os.remove("tmp")



if __name__ == "__main__":
	pars = ArgumentParser()
	pars.add_argument('--target',
	                  type=str,
	                  help='path to target mesh')
	pars.add_argument('--model',
	                  type=str,
	                  help='path to test data')
	pars.add_argument('--weights',
	                  type=str,
	                  help='path to model weights')

	args = pars.parse_args()
	if args.target == None:
		raise Exception("Target file was not specified")

	if args.model == None:
		raise Exception("Path to model was not specified")

	if args.weights == None:
		raise Exception("Path to weights was not specified")

	main(pars.parse_args())