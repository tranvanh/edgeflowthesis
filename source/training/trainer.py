import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.loader import DataLoader

from training.gnn_model import GNN
from utils.io_operations import save_model, create_dir

if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

class Trainer(object):
	def __init__(self, model, train_data, test_data, epochs):
		self.model = model
		self.train_loader = DataLoader(train_data)
		self.test_loader = DataLoader(test_data)
		self.epochs = epochs
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

	def train_loop(self, current_min):
		train_loss = 0
		size = len(self.train_loader)
		loop = tqdm(range(size))

		# Set the model to training mode - important for batch normalization and dropout layers
		self.model.train()
		for i in loop:
			train_features, train_labels = next(iter(self.train_loader))
			self.optimizer.zero_grad()
			out = self.model(train_features)
			loss = F.mse_loss(out, train_labels[0])
			loss.backward()
			self.optimizer.step()
			loop.set_description('loss = %.6f' % loss)
			train_loss += loss.item()
			
		avg_train_loss = train_loss/size
		if avg_train_loss < current_min:
			current_min = avg_train_loss
			save_model(self.model, "model/best_model")
			print("SAVED")
		print(f"Train Error: Avg loss: {avg_train_loss:>8f}")
		return current_min

	def train(self):
		print("Model training started")
		current_min =  float('inf')
		create_dir("model")
		for i in range(self.epochs):
			print(f"EPOCH {i+1}/{self.epochs}")
			current_min = self.train_loop(current_min)

	def evaluate(self):
		print("Model evaluation started")
		self.model.eval()
		size = len(self.test_loader.dataset)
		num_batches = len(self.test_loader)
		test_loss, correct = 0, 0

		with torch.no_grad():
			for X, y in self.test_loader:
				out = self.model(X)
				test_loss += F.mse_loss(out, y[0]).item()

		test_loss /= num_batches