import torch
import numpy as np

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def mesh_loss(predicted, gt, sample_volume, w_normal, w_laplacian):
    w_chamfer = 1.0
    w_edge = 1.0
    
    sample_trg = sample_points_from_meshes(gt, sample_volume)
    sample_src = sample_points_from_meshes(predicted, sample_volume)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(predicted)

    # mesh normal consistency
    loss_normal = mesh_normal_consistency(predicted)

    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(predicted, method="uniform")

    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian 

    return loss

def fit_src_to_trg(file_name, src_mesh, trg_mesh):
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=0.001)

    # Number of optimization steps
    Niter = 80000

    loop = tqdm(range(Niter))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

       # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)

        # adaptive weights, we want to get the the general shape quickly
        w_normal = 0.1
        w_laplacian = 1.0
        if i < 20000:
            w_laplacian = 1.0
        else:
            w_laplacian = 0.5
        
        loss = mesh_loss(new_src_mesh, trg_mesh, 3*trg_mesh.verts_packed().shape[0], w_normal, w_laplacian)
        
        # Print the losses
        loop.set_description(f"{file_name} total_loss = {loss}.6f")

        # Optimization step
        loss.backward()
        optimizer.step()
    
    return deform_verts

