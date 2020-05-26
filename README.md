# Autoencoder to Classifier

In this repository i recreated the methods used in the paper "What do Deep Networks Like to See?" https://arxiv.org/abs/1803.08337 

1. Train a autoencoder architecture.
1. Finetune classifier.
1. Combine autoencoder and classifier.
    1. Freeze weights of encoder and classifier.
    1. Finetune combined decoder.
1. Create reconstructed images with finetuned model and plot histogram to compare different autoencoders

All code was created using Tensorflow 1.14.

Weights must first be unzipped to be loaded.
