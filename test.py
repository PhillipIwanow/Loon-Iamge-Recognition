# Core libraries
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50
from models.TripletResnet import TripletResnet50
from models.TripletResnetSoftmax import TripletResnet50Softmax
from models.SimpleConvNet import SimpleConvNetPrototype



# Import our dataset class
from datasets.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if you have else use cpu

# For a trained model, let's evaluate it
def evaluateModel(args):
	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, True)
	test_dataset = Utilities.selectDataset(args, False)

	# Get the embeddings and labels of the training set and testing set
	train_embeddings, train_labels = inferEmbeddings(args, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, test_dataset, "test")

	# Classify them
	accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)

	# Write it out to the console so that subprocess can pick them up and close
	sys.stdout.write(f"Accuracy={str(accuracy)}")
	sys.stdout.flush()
	sys.exit(0)

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Sanity check for NaNs in the input embeddings
    print("Checking for NaNs in train/test embeddings...")
    print("NaNs in train embeddings:", np.isnan(train_embeddings).sum())
    print("NaNs in test embeddings:", np.isnan(test_embeddings).sum())
    print("NaNs in train labels:", np.isnan(train_labels).sum())
    print("NaNs in test labels:", np.isnan(test_labels).sum())

    # Replace NaNs with zeros (temporary fix; better to fix at source)
    if np.isnan(train_embeddings).any() or np.isnan(test_embeddings).any():
        print("WARNING: NaNs detected in embeddings! Replacing with zeros.")
        train_embeddings = np.nan_to_num(train_embeddings)
        test_embeddings = np.nan_to_num(test_embeddings)
    if np.isnan(train_labels).any() or np.isnan(test_labels).any():
        print("WARNING: NaNs detected in labels! Replacing with zeros.")
        train_labels = np.nan_to_num(train_labels)
        test_labels = np.nan_to_num(test_labels)

    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

    # Fit the KNN
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels)

    # Get predictions
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    return accuracy


# Infer the embeddings for a given dataset

def inferEmbeddings(args, dataset, split):
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

    # Select model based on args.model
    if args.model == "TripletResnetSoftmax":
        model = TripletResnet50Softmax(pretrained=False, num_classes=dataset.getNumClasses())
    elif args.model == "TripletResnet":
        model = TripletResnet50(pretrained=False, num_classes=dataset.getNumClasses())
    elif args.model == "SimpleConvNet":
        model = SimpleConvNetPrototype(num_classes=dataset.getNumClasses(), embedding_size=args.embedding_size)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Load checkpoint weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    model.to(device)
    model.eval()

    outputs_embedding = np.zeros((1, args.embedding_size))
    labels_embedding = np.zeros((1))

    # Inference loop
    for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
        images = Variable(images.to(device))

        # For SimpleConvNet, use forward_sibling; for others, use standard forward
        if args.model == "SimpleConvNet":
            embeddings, _ = model.forward_sibling(images)
        else:
            # ResNet-based models: output is the embedding
            embeddings = model(images)

        embeddings = embeddings.data
        embeddings = embeddings.cpu().numpy()

        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()

        outputs_embedding = np.concatenate((outputs_embedding, embeddings), axis=0)
        labels_embedding = np.concatenate((labels_embedding, labels), axis=0)

    if args.save_embeddings:
        save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")
        np.savez(save_path, embeddings=outputs_embedding, labels=labels_embedding)

    return outputs_embedding, labels_embedding


# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Params')

    # Required arguments
    parser.add_argument('--model', type=str, default='TripletResnetSoftmax',
                        help='Which model to use: [TripletResnetSoftmax, TripletResnet, SimpleConvNet]')
    parser.add_argument('--model_path', nargs='?', type=str, required=True,
                        help='Path to the saved model to load weights from')
    parser.add_argument('--folds_file', type=str, default="", required=True,
                        help="The file containing known/unknown splits")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Where to store the embeddings")

    parser.add_argument('--dataset', nargs='?', type=str, default='OpenSetCows2020',
                        help='Which dataset to use')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128,
                        help='Size of the dense layer for inference')
    parser.add_argument('--current_fold', type=int, default=0,
                        help="The current fold we'd like to test on")
    parser.add_argument('--save_embeddings', type=bool, default=True,
                        help="Should we save the embeddings to file")
    args = parser.parse_args()

    # Let's infer some embeddings
    evaluateModel(args)
