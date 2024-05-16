import pandas as pd
import torch
from torch import nn
from feature_selection import *
import numpy as np
from sklearn.model_selection import train_test_split

HIDDEN_LAYER_SIZE = 128
BATCHES = 50
EPOCHS = 1000
LEARNING_RATE = 0.002


class NeuralNetwork(nn.Module):
    def __init__(self, sample_instance):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(len(sample_instance), HIDDEN_LAYER_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_LAYER_SIZE, 5)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

    def predict_label(self, instance):
        logits = self(instance)
        predicted_probabilities = nn.Softmax(dim=0)(logits)
        return predicted_probabilities


def train_loop(model, features, labels, loss_fn, optimiser):
    model.train()

    for i in range(len(features)):
        pred = model(features[i])
        loss = loss_fn(pred, labels[i])

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


def test_loop(model, features, labels, loss_fn, size):
    model.eval()
    num_batches = len(features)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i in range(len(features)):
            pred = model(features[i])
            test_loss += loss_fn(pred, labels[i]).item()
            correct += (pred.argmax(1) == labels[i]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def to_tensor(features, labels, device):
    feature_batches = np.array_split(features, BATCHES)
    label_batches = np.array_split(labels, BATCHES)

    for i in range(len(feature_batches)):
        feature_batches[i] = torch.from_numpy(feature_batches[i].to_numpy(dtype=np.float32)).to(device)
        label_batches[i] = torch.from_numpy(label_batches[i].to_numpy(dtype=np.float32)).to(device)

    return feature_batches, label_batches


def neural_net(train_df):
    train_df_labels = train_df['imdb_score_binned']
    train_df_features = train_df.drop('imdb_score_binned', axis=1)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = NeuralNetwork(train_df_features.iloc[0]).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    feature_batches, label_batches = to_tensor(train_df_features, train_df_labels, device)

    for i in range(EPOCHS):
        train_loop(model, feature_batches, label_batches, loss_fn, optimiser)
        test_loop(model, feature_batches, label_batches, loss_fn, len(train_df_features))

    return model, device


def run_neural_network(model, device, test):
    test_tensor = torch.from_numpy(test.to_numpy(dtype=np.float32)).to(device)
    pred = model.predict_label(test_tensor).argmax(1).tolist()

    test = test.copy()
    test['imdb_score_binned'] = pred

    test.to_csv('CSVs/neural_network.csv', columns=['id', 'imdb_score_binned'], index_label=False)

    return test


def validate():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()

    X, y = train_test_split(train_df_std, train_size=0.9)

    result = neural_net(X, y.drop('imdb_score_binned', axis=1))
    accuracy = np.sum(result['imdb_score_binned'] == y['imdb_score_binned']) / y.shape[0]

    return accuracy

    # test_df_std.to_csv("CSVs/neural_network.csv", columns=['id', 'imdb_score_binned'], index=False)


def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    result = neural_net(train_df_std, test_df_std)
    result.to_csv("CSVs/neural_network.csv", columns=['id', 'imdb_score_binned'], index=False)


if __name__ == '__main__':
    main()
