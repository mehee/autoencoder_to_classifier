# Autoencoder to Classifier

In this repository i recreated the methods used in the paper "What do Deep Networks Like to See?" https://arxiv.org/abs/1803.08337 

1. Train a autoencoder architecture.
1. Finetune classifier for your dataset.
1. Combine autoencoder and classifier.
    1. Freeze weights of encoder and classifier.
    1. Finetune combined models.
1. Create reconstructed images with finetuned autoencoder and plot histogram to compare different autoencoders

Weights for the models can be found here: https://drive.google.com/drive/folders/1rIyXKn9We3dr8zgq3lAWeuBVsfcfTBgf
