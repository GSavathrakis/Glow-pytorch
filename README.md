# Glow-pytorch
This is a pytorch implementation of the Glow Normalizing Flow model as proposed in the paper "[Glow: Generative Flow
with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039)" by Kingma et al.

It can either be used for training or inference on a pretrained model.

For training, an example for the MNIST dataset is to run: 
`python main.py train --dataset MNIST`

For inference run:
`python main.py generate --model_path path_to_model.pt`

In the generate.py script one has to manually change the data shape depending on the dataset on which the model was trained. The one used is (1,28,28) which is for the MNIST dataset.
