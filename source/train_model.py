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

def main(args):
	model = GNN(6,256,3).to(device)
	trainer = Trainer(model, args.epochs)
	if not args.skip_training:
		print("Loading training data")
		train_data = CustomDataset(args.train_data)
		trainer.train(train_data)
	else:
		print("Skipping training")
	if args.evaluate:
		print("Loading test data")
		test_data = CustomDataset(args.test_data)
		trainer.evaluate(test_data)

if __name__ == "__main__":
	pars = ArgumentParser()
	pars.add_argument('--train-data',
					  type=str,
					  default="data/train",
					  help='[path to train data]{/data/train}')
	pars.add_argument('--test-data',
					  type=str,
					  default="data/test",
					  help='[path to test data]{data/test}')
	pars.add_argument('--epochs',
					  type=int,
					  default=500,
					  help='[path to model weights]{500}')
	pars.add_argument('--skip-training',
					  type=bool,
					  default=False,
					  help='[skiping training the model]{False}')
	pars.add_argument('--evaluate',
					  type=bool,
					  default=True,
					  help='[evaluate the model]{True}')

	main(pars.parse_args())