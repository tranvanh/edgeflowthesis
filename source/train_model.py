import torch
from argparse import ArgumentParser

from training.dataset import CustomDataset
from training.gnn_model import GNN
from training.trainer import Trainer

if torch.cuda.is_available():
	print("CUDA avaliable")
	device = torch.device("cuda:0")
else:
	print("CUDA unavailable")
	device = torch.device("cpu")


# load checkpoint
def main(args):

	train_path = "data/train" if args.train_data == None else args.train_data
	test_path = "data/test" if args.test_data == None else args.test_data
	epochs = 500 if args.epochs == None else args.epochs

	model = GNN(6,256,3).to(device)

	# load checkpoint weights if available
	if not args.weights == None:
		model.load_state_dict(torch.load(args.weights))

	print("Loading training data")
	train_data = CustomDataset(train_path)
	print("Loading test data")
	test_data = CustomDataset(test_path)
	print("Data loaded")

	trainer = Trainer(model, train_data, test_data, epochs)
	trainer.train()
	trainer.evaluate()

if __name__ == "__main__":
	pars = ArgumentParser()
	pars.add_argument('--train-data',
	                  type=str,
	                  help='path to train data')
	pars.add_argument('--test-data',
	                  type=str,
	                  help='path to test data')
	pars.add_argument('--weights',
	                  type=str,
	                  help='path to model weights')
	pars.add_argument('--epochs',
	                  type=int,
	                  help='path to model weights')

	main(pars.parse_args())