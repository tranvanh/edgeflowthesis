import os
import torch
import numpy as np
from utils.io_wrappers import load_obj_wrapper, load_ply_wrapper, save_tensor, save_obj_wrapper
from utils.mesh_utils import gen_sphere, decimate_mesh
from deformation.deform_mesh import fit_src_to_trg

from os.path import isfile, join
from pytorch3d.structures import Meshes, Pointclouds

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA available")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

dir_path = "shrec"
def create_dir(file_name):
    new_path = f"{dir_path}/prep/{file_name[:-4]}"
    if not os.path.exists(new_path):
        print(f"directory created for {file_name}")
        os.makedirs(new_path)
    else:
        print(f"directory exists for {file_name}")
    return new_path

def main():
    files = os.listdir(dir_path)
    for f in files:
        if f == "prep":
            continue
        
        new_path = create_dir(f)
        trg_mesh, scale , center = load_obj_wrapper(f"{dir_path}/{f}")

        if(trg_mesh.verts_packed().shape[0] > 10000):
            decimate_mesh(f"{dir_path}/{f}")
            trg_mesh, scale , center = load_obj_wrapper(f"{dir_path}/{f}")
        
        trg_torch = trg_mesh.verts_packed()
        gen_sphere(trg_torch.shape[0], new_path)
        src_mesh, _, _ = load_ply_wrapper(f"{new_path}/sphere.ply")
        deform_offsets = fit_src_to_trg(f, src_mesh, trg_mesh)

        save_obj_wrapper(src_mesh.offset_verts(deform_offsets), f"{new_path}/deformed.obj", scale, center)
        save_tensor(deform_offsets, f"{new_path}/offsets.pt")

if __name__ == "__main__":
    main()