import numpy as np
import pyvista as pv
import open3d as o3d

"""
Generate evenly distributed sphere

/source https://stackoverflow.com/questions/47485235/i-want-to-make-evenly-distributed-sphere-in-vtk-python 
"""
def gen_sphere(num_points, path):
	indices = np.arange(0, num_points, dtype=float) + 0.5
	
	phi = np.arccos(1 - 2*indices/num_points)
	theta = np.pi * (1 + 5**0.5) * indices
	
	x =  np.cos(theta) * np.sin(phi) 
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(phi)
	
	point_cloud = pv.PolyData(np.c_[x, y, z])
	surface = point_cloud.delaunay_3d().extract_surface()
	surface.save(f"{path}/sphere.ply")

def decimate_mesh(file_name, max_triangle_count):
	mesh = o3d.io.read_triangle_mesh(file_name)
	simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_triangle_count)
	# Save the simplified mesh
	o3d.io.write_triangle_mesh(file_name, simplified_mesh)