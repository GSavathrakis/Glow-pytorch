import argparse
import warnings

from train import train
from generate import generate

def parser():
	parser = argparse.ArgumentParser(description="Glow Normalizing Flow model Parameters")
	subparsers = parser.add_subparsers(help="The mode of the program", dest="mode")
	parser_train = subparsers.add_parser("train", help="Train the Glow model")
	parser_generate = subparsers.add_parser("generate", help="Use trained Glow model to generate samples")
	
	parser_train.add_argument('--dataset', type=str, default='MNIST')
	parser_train.add_argument('--batch_size', type=int, default=128)
	parser_train.add_argument('--conditional', type=bool, default=False)
	parser_train.add_argument('--lr', type=float, default=1e-3)
	parser_train.add_argument('--weight_decay', type=float, default=1e-4)
	parser_train.add_argument('--epochs', type=int, default=10)
	parser_train.add_argument('--hidden_channels', type=int, default=1000)
	parser_train.add_argument('--num_flow_steps', type=int, default=10)
	parser_train.add_argument('--ACL_layers', type=int, default=10)
	parser_train.add_argument('--num_levels', type=int, default=1)
	parser_train.add_argument('--num_splits', type=int, default=2)
	parser_train.add_argument('--chunk_size', type=int, default=1)


	parser_generate.add_argument("--model_path", type=str, help="The path to the saved model")
	parser_generate.add_argument('--dataset', type=str, default='MNIST')
	parser_generate.add_argument('--conditional', type=bool, default=False)
	parser_generate.add_argument("--num_samples", type=int, default=15, help="Number of samples to be generated")
	parser_generate.add_argument('--hidden_channels', type=int, default=1000)
	parser_generate.add_argument('--num_flow_steps', type=int, default=10)
	parser_generate.add_argument('--ACL_layers', type=int, default=10)
	parser_generate.add_argument('--num_levels', type=int, default=1)
	parser_generate.add_argument('--num_splits', type=int, default=2)
	parser_generate.add_argument('--chunk_size', type=int, default=1)

	return parser.parse_args()

def main():
	args = parser()
	if (args.mode=='train'):
		train(args)
	elif (args.mode=='generate'):
		generate(args)

if __name__ == "__main__":
	main()