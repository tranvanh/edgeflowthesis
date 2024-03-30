import torch
from model.gnn_models import GNN
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class Trainer(object):
    def __init__(self, data):
        self.model = GNN(3,3).to(device)

    def train():
    	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    	iter = 5000
		loop = tqdm(range(Niter))

		model.train()
		for i in loop:
		    optimizer.zero_grad()
		    out = model(data)
		    # new_src_mesh = sphere_mesh.offset_verts(out)
		    loss = F.mse_loss(out, deform_verts)
		    # loss = meshLoss(out, trg_mesh)
		    loop.set_description('total_loss = %.6f' % loss)
		    loss.backward()
		    optimizer.step()