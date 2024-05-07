import torch
from training.dataset import CustomDataset
from training.gnn_model import GNN
from training.trainer import Trainer

train_path = "data/train"
test_path = "data/test"

if torch.cuda.is_available():
	print("CUDA avaliable")
	device = torch.device("cuda:0")
else:
	print("CUDA unavailable")
	device = torch.device("cpu")

def main():
	model = GNN(6,256,3).to(device)
	print("Loading training data")
	train_data = CustomDataset(train_path)
	print("Loading test data")
	test_data = CustomDataset(test_path)
	print("Data loaded")
	epochs = 500

	trainer = Trainer(model, train_data, test_data, epochs)
	trainer.train()
	trainer.evaluate()

if __name__ == "__main__":
	main()