import os
import torch
import numpy as np
import utils.io_wrappers
import deformation.deform_mesh

from os.path import isfile, join
from pytorch3d.structures import Meshes, Pointclouds
from model.gnn_models import GNN

dir_path = "shrec/prep"

def main():
    files = os.listdir(dir_path)
    for f in files:

        # load trg_mesh, offsets, 

        new_path = create_dir(f)
        trg_mesh, scale , center = load_obj_wrapper(f"{dir_path}/{f}")
        trg_torch = trg_mesh.verts_packed()
        
        gen_sphere(trg_torch.shape[0], new_path)
        src_mesh = load_ply_wrapper(f"{new_path}/sphere.ply")
        deform_offsets = fit_src_to_trg(f, src_mesh, trg_mesh)

        save_obj_wrapper(src_mesh.offset_verts(deform_offsets), new_path, scale, center)
        save_tensor(deform_offsets, "f{new_path}/offsets.pt")

if __name__ == "__main__":
    main()