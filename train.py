import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tslearn.datasets import UCR_UEA_datasets
from models.ResNet import ResNetBaseline
from tsai.models.InceptionTime import InceptionTime
from utils.utils import create_path_if_not_exists, evaluate_classifier
import argparse


parser = argparse.ArgumentParser(description='Train a target model.')
parser.add_argument('-m', '--model_type', type=str, default='InceptionTime', choices=['ResNet', 'InceptionTime'])
parser.add_argument('-d', '--dataset', type=str, default='GunPoint', help="Datasets are loaded from UCR UEA Repository.")
parser.add_argument('-p', '--base_path', type=str, default='saved_models', help="Base path to save models.")
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate.')
args = parser.parse_args()

np.random.seed(0)

if __name__ == '__main__':
    dataset_name = args.dataset
    model_type = args.model_type
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    base_path = args.base_path

    if dataset_name == 'AudioMNIST':
        x_train_cpu = np.load("datasets/AudioMNIST/AudioNet_digit_0_x_train.npy")
        y_train_cpu = np.load("datasets/AudioMNIST/AudioNet_digit_0_y_train.npy")
        x_test_cpu = np.load("datasets/AudioMNIST/AudioNet_digit_0_x_test.npy")
        y_test_cpu = np.load("datasets/AudioMNIST/AudioNet_digit_0_y_test.npy")
        # Down-sampling the audio signal
        x_train_cpu = x_train_cpu[:, :, 0, 0, ::2]
        y_train_cpu = y_train_cpu[:, 0, 0]
        x_test_cpu = x_test_cpu[:, :, 0, 0, ::2]
        y_test_cpu = y_test_cpu[:, 0, 0]
    else:
        uea_ucr = UCR_UEA_datasets(use_cache=True)
        x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu = uea_ucr.load_dataset(dataset_name)
        x_train_cpu = np.swapaxes(x_train_cpu, 1, 2)
        x_test_cpu = np.swapaxes(x_test_cpu, 1, 2)

    # There might be datasets where class labels do not start from 0
    if np.min(y_train_cpu) > 0:
        num_classes = np.max(y_train_cpu)
        y_train_cpu = y_train_cpu - 1
        y_test_cpu = y_test_cpu - 1
    elif np.min(y_train_cpu) < 0:
        num_classes = np.max(y_train_cpu)+1
        y_train_cpu[y_train_cpu==-1] = 0
        y_test_cpu[y_test_cpu==-1] = 0
    else:
        num_classes = np.max(y_train_cpu)+1

    features = x_train_cpu.shape[1]
    steps = x_train_cpu.shape[2]

    x_train = torch.tensor(x_train_cpu).float().cuda()
    y_train = F.one_hot(torch.tensor(y_train_cpu)).float().cuda()
    x_test = torch.tensor(x_test_cpu).float().cuda()
    y_test = F.one_hot(torch.tensor(y_test_cpu)).float().cuda()

    if dataset_name == "GestureMidAirD1":
        x_train_cpu = np.nan_to_num(x_train_cpu, nan=0)
        x_test_cpu = np.nan_to_num(x_test_cpu, nan=0)

    model_path = base_path + "/" + dataset_name + "/" + model_type + "/"
    create_path_if_not_exists(model_path)
    model_path_full = model_path + "state_dict_model.pt"

    if model_type == "ResNet":
        model = ResNetBaseline(in_channels=features, mid_channels=32, num_pred_classes=num_classes).float().cuda()
    elif model_type == "InceptionTime":
        model = InceptionTime(c_in=features, c_out=num_classes, seq_len=None, nf=32, nb_filters=None).float().cuda()

    train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(x_test, y_test), shuffle=False, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accuracy measurement
            y_pred = torch.argmax(y_pred.data, 1)
            y_lbl = torch.argmax(y_batch, 1)
            correct += (y_pred == y_lbl).sum()
        train_acc = correct / len(train_loader.dataset)
        # Validation
        model.eval()
        with torch.no_grad():
            test_acc = evaluate_classifier(model, test_loader)
        print("Epoch {:d}: train/test accuracy is {:.2f}/{:.2f}.".format(epoch, 100 * train_acc, 100 * test_acc))

    print("Training has finished.")
    torch.save(model.state_dict(), model_path_full)

