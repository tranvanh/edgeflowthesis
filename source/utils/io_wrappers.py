import torch
import os

from pytorch3d.io import load_obj, save_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # print("CUDA available")
else:
    device = torch.device("cpu")
    # print("WARNING: CPU only, this will be slow!")


def create_dir(file_path):
    if not os.path.exists(file_path):
        print(f"directory at \'{file_path}\'created")
        os.makedirs(file_path)
    else:
        print(f"directory at \'{file_path}\' already exists")
    return file_path

# We read the target 3D model using load_obj
def load_obj_wrapper(fileName):
    verts, faces, aux = load_obj(fileName)
    
    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    
    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    
    # We construct a Meshes structure for the target mesh
    #print(aux.normals.shape)
    return Meshes(verts=[verts], faces=[faces_idx]), scale, center


def load_ply_wrapper(fileName):
    verts, faces = load_ply(fileName)
    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces = faces.to(device)
    verts = verts.to(device)
    
    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    
    return Meshes(verts = [verts], faces = [faces]), scale, center

def save_obj_wrapper(mesh, fileName, scale=1.0, center =0.0):
    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = mesh.get_mesh_verts_faces(0)
    
    # Scale normalize back to the original target size
    final_verts = final_verts * scale + center
    
    # Store the predicted mesh using save_obj
    save_obj(fileName, final_verts, final_faces)


def save_tensor(obj, path):
    torch.save(obj, path)

def load_tensor(path):
    torch.load(path)

def save_model(model, name):
    torch.save(model, f"{name}.pth")
    torch.save(model.state_dict(), f"{name}_weights.pth")

def load_model(model_path, weight_path):
    model = torch.load(path)
    model.load_state_dict(torch.load(weight_path))
    return model