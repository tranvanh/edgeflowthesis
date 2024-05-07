import os
import torch
import numpy as np
from argparse import ArgumentParser
from pytorch3d.structures import Meshes, Pointclouds
from os.path import isfile, join
from utils.io_operations import (
	create_dir,
	load_obj_wrapper,
	load_ply_wrapper,
	save_tensor,
	save_obj_wrapper
)
from utils.mesh_utils import gen_sphere, decimate_mesh
from deformation.deform_mesh import fit_src_to_trg

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("CUDA available")
else:
	device = torch.device("cpu")
	print("CUDA unavailable")

def main(args):
	dir_path = args.data_path
	files = os.listdir(dir_path)
	for f in files:
		if f == "prep":
			continue
		
		new_path = f"{dir_path}/prep/{f[:-4]}"
		create_dir(new_path)
		trg_mesh, scale , center = load_obj_wrapper(f"{dir_path}/{f}")

		if(trg_mesh.verts_packed().shape[0] > args.point_limits):
			decimate_mesh(f"{dir_path}/{f}", args.point_limits)
			trg_mesh, scale , center = load_obj_wrapper(f"{dir_path}/{f}")
		
		trg_torch = trg_mesh.verts_packed()
		gen_sphere(trg_torch.shape[0], new_path)
		src_mesh, _, _ = load_ply_wrapper(f"{new_path}/sphere.ply")
		deform_offsets = fit_src_to_trg(f, src_mesh, trg_mesh, args.iterations, args.learning_rate)

		save_obj_wrapper(src_mesh.offset_verts(deform_offsets), f"{new_path}/deformed.obj", scale, center)
		save_tensor(deform_offsets, f"{new_path}/offsets.pt")

if __name__ == "__main__":
	pars = ArgumentParser()
	pars.add_argument('--data-path',
					  type=str,
					  default="shrec",
					  help='path to train data')
	pars.add_argument('--iterations',
					  type=int,
					  default=80000,
					  help='number of iterations')
	pars.add_argument('--point-limits',
					  type=int,
					  default=10000,
					  help='maximal number of points')
	pars.add_argument('--learning-rate',
					  type=float,
					  default=0.001,
					  help='minimization rate')
	main(pars.parse_args())