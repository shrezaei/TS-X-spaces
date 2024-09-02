import os.path
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import torch.utils.data as data
from tslearn.datasets import UCR_UEA_datasets
from captum.attr import Occlusion, InputXGradient, IntegratedGradients, Saliency, GradientShap, DeepLift, Lime, KernelShap, GuidedBackprop
from models.ResNet import ResNetBaseline
from tsai.models.InceptionTime import InceptionTime
from utils.utils import create_path_if_not_exists, PAF, explain, evaluate_classifier
from utils.utils import Robustness
from utils.utils import sparsity
from utils.wrappers import FFT_Wrapper, Spectrogram_Wrapper, Diff_Wrapper, Min_Zero_Wrapper
from utils.plot_attribution import plot_attribution
from captum.attr import visualization
import argparse


parser = argparse.ArgumentParser(description='Explain the target model.')
parser.add_argument('-m', '--model_type', type=str, default='ResNet', choices=['ResNet', 'InceptionTime'])
parser.add_argument('-a', '--xai_approach', type=str, default='DeepLift', choices=['DeepLift', 'IntegratedGradients', 'InputXGradient',
                                                                                   'Saliency', 'GradientShap', 'Lime', 'Occlusion', 'KernelShap',
                                                                                   'GuidedBackprop'])
parser.add_argument('-d', '--dataset', type=str, default='GunPoint', help="Datasets are loaded from UCR UEA Repository.")
parser.add_argument('-p', '--base_path', type=str, default='saved_models', help="Base path to save models.")
parser.add_argument('-x', '--X_space', type=str, default='Time', choices=['Time', 'Freq', 'TimeFreq', 'Diff', 'MinZero', 'Diff_back_to_Time'])
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('-i', '--sample_id', type=int, default=0, help='Sample number from the dataset.')
parser.add_argument('-s', '--set', type=str, default='Train', choices=['Train', 'Test'])
parser.add_argument('-n', '--nfft', type=int, default=5, help='Spectrogram (SFFT) parameter.')
args = parser.parse_args()

np.random.seed(0)

if __name__ == '__main__':
    dataset_name = args.dataset
    model_type = args.model_type
    batch_size = args.batch_size
    base_path = args.base_path
    x_space = args.X_space
    id = args.sample_id
    set = args.set
    xai_name = args.xai_approach
    nfft = args.nfft

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

    train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(x_test, y_test), shuffle=False, batch_size=batch_size)

    if os.path.exists(model_path_full):
        model.load_state_dict(torch.load(model_path_full))
        model.eval()
        acc_train = evaluate_classifier(model, train_loader)
        acc_test = evaluate_classifier(model, test_loader)
        print("The model with train/test accuracy of {:.2f}/{:.2f} was loaded.\n".format(100*acc_train, 100*acc_test))
    else:
        raise NameError("The model does not exist.")

    if xai_name == 'DeepLift':
        xai_ref = DeepLift
    elif xai_name == 'IntegratedGradients':
        xai_ref = IntegratedGradients
    elif xai_name == 'InputXGradient':
        xai_ref = InputXGradient
    elif xai_name == 'Saliency':
        xai_ref = Saliency
    elif xai_name == 'GradientShap':
        xai_ref = GradientShap
    elif xai_name == 'Lime':
        xai_ref = Lime
    elif xai_name == 'Occlusion':
        xai_ref = Occlusion
    elif xai_name == 'KernelShap':
        xai_ref = KernelShap
    elif xai_name == 'GuidedBackprop':
        xai_ref = GuidedBackprop

    if x_space == "Time":
        explain_model = model
        xai = xai_ref(explain_model)
        if set == 'Train':
            target_sample = x_train[id:id+1]
        else:
            target_sample = x_test[id:id+1]

        predicted_label = torch.argmax(explain_model(target_sample)).item()
        exp = explain(xai, xai_name, target_sample, predicted_label, sliding_window=(1, 5), baselines=x_train[-40:])

        classifier_robustness, xai_robustness = Robustness(target_sample, exp, explain_model, xai, xai_name, x_space, sliding_window=(1, 5), baselines=x_train[-40:])
        paf = PAF(target_sample, exp, explain_model, predicted_label, type=x_space, epsilon=0.01)    #-1 means unsuccessful explanation

        exp = exp.detach().cpu().clone().numpy()
        plot_attribution(np.array([target_sample.cpu().numpy()]), exp, figsize=(12, 6), title="Time")

    elif x_space == "Freq":
        explain_model = FFT_Wrapper(model, steps).float().cuda()
        explain_model.eval()
        xai = xai_ref(explain_model)

        temp = torch.fft.rfft(x_train, dim=2)
        x_train_fft = torch.tensor(torch.concatenate((torch.log(torch.abs(temp) + 1), torch.angle(temp)), dim=1)).float().cuda()
        temp = torch.fft.rfft(x_test, dim=2)
        x_test_fft = torch.tensor(torch.concatenate((torch.log(torch.abs(temp) + 1), torch.angle(temp)), dim=1)).float().cuda()
        if set == 'Train':
            target_sample = x_train_fft[id:id+1]
        else:
            target_sample = x_test_fft[id:id+1]

        predicted_label = torch.argmax(explain_model(target_sample)).item()
        exp = explain(xai, xai_name, target_sample, predicted_label, sliding_window=(1, 5), baselines=x_train[-40:])

        classifier_robustness, xai_robustness = Robustness(target_sample, exp, explain_model, xai, xai_name, x_space, sliding_window=(1, 5), baselines=x_train[-40:])
        paf = PAF(target_sample, exp, explain_model, predicted_label, type=x_space, epsilon=0.01)    #-1 means unsuccessful explanation

        exp = exp.detach().cpu().clone().numpy()
        plot_attribution(np.array([target_sample[:, 0:1, :].cpu().numpy()]), exp[:, 0:1, :], figsize=(12, 6), title="Frequency")
        plot_attribution(np.array([target_sample[:, 1:2, :].cpu().numpy()]), exp[:, 1:2, :], figsize=(12, 6), title="Phase")

    elif x_space == 'TimeFreq':
        explain_model = Spectrogram_Wrapper(model, n_fft=nfft).float().cuda()
        explain_model.eval()
        xai = xai_ref(explain_model)

        spec_function = torchaudio.transforms.Spectrogram(power=None, n_fft=nfft, hop_length=int(nfft/2)).cuda()
        temp = spec_function(x_train)
        x_train_spec = torch.concatenate((torch.abs(temp), torch.angle(temp)), dim=1).float().cuda()
        temp = spec_function(x_test)
        x_test_spec = torch.concatenate((torch.abs(temp), torch.angle(temp)), dim=1).float().cuda()
        if set == 'Train':
            target_sample = x_train_spec[id:id+1]
        else:
            target_sample = x_test_spec[id:id+1]

        predicted_label = torch.argmax(explain_model(target_sample)).item()
        exp = explain(xai, xai_name, target_sample, predicted_label, sliding_window=(1, 1, 1), baselines=x_train[-40:])

        classifier_robustness, xai_robustness = Robustness(target_sample, exp, explain_model, xai, xai_name, x_space, sliding_window=(1, 1, 1), baselines=x_train[-40:])
        paf = PAF(target_sample, exp, explain_model, predicted_label, type=x_space,
                  epsilon=0.01)  # -1 means unsuccessful explanation

        exp = exp.detach().cpu().clone().numpy()

        exp_abs = np.moveaxis(exp[0, 0:1], 0, -1)
        img_abs = np.moveaxis(target_sample[:, 0:1].cpu().numpy(), 0, -1)
        if np.sum(exp_abs > 0):  # Particularly for LIME, it's so bad it may give zero attribution to all time/frequecy components.
            visualization.visualize_image_attr(exp_abs, img_abs, "blended_heat_map", sign="all", alpha_overlay=0.3, show_colorbar=True, fig_size=(9, 6))

    elif x_space == 'MinZero':
        explain_model = Min_Zero_Wrapper(model, input_size=steps).float().cuda()
        explain_model.eval()
        xai = xai_ref(explain_model)

        x_train_min_zero_cpu = np.concatenate((np.copy(x_train_cpu), np.zeros((x_train_cpu.shape[0], features, 1))), axis=2)
        x_test_min_zero_cpu = np.concatenate((np.copy(x_test_cpu), np.zeros((x_test_cpu.shape[0], features, 1))), axis=2)
        x_train_min_zero_cpu[:, :, -1] = np.min(x_train_min_zero_cpu[:, :, :-1], axis=2)
        x_test_min_zero_cpu[:, :, -1] = np.min(x_test_min_zero_cpu[:, :, :-1], axis=2)
        for i in range(steps):
            x_train_min_zero_cpu[:, :, i] -= x_train_min_zero_cpu[:, :, -1]
            x_test_min_zero_cpu[:, :, i] -= x_test_min_zero_cpu[:, :, -1]
        x_train_min_zero = torch.tensor(x_train_min_zero_cpu).float().cuda()
        x_test_min_zero = torch.tensor(x_test_min_zero_cpu).float().cuda()
        if set == 'Train':
            target_sample = x_train_min_zero[id:id+1]
        else:
            target_sample = x_test_min_zero[id:id+1]

        predicted_label = torch.argmax(explain_model(target_sample)).item()
        exp = explain(xai, xai_name, target_sample, predicted_label, sliding_window=(1, 5), baselines=x_train[-40:])

        # We ignore the last step because it stores the min value. Unlike each time step, it has a global effect if changed
        # Hence algorithm often assign high attribution to it.
        exp[:, :, -1] = 0

        classifier_robustness, xai_robustness = Robustness(target_sample, exp, explain_model, xai, xai_name, x_space, sliding_window=(1, 5), baselines=x_train[-40:])
        paf = PAF(target_sample, exp, explain_model, predicted_label, type=x_space, epsilon=0.01)  # -1 means unsuccessful explanation

        exp = exp.detach().cpu().clone().numpy()
        plot_attribution(np.array([target_sample.cpu().numpy()]), exp, figsize=(12, 6), title="Mean Zero Space")
    elif x_space in ['Diff', 'Diff_back_to_Time']:
        explain_model = Diff_Wrapper(model, input_size=steps).float().cuda()
        explain_model.eval()
        xai = xai_ref(explain_model)

        x_train_diff_cpu = np.copy(x_train_cpu)
        x_test_diff_cpu = np.copy(x_test_cpu)
        x_train_diff_cpu[:, :, 1:] = x_train_cpu[:, :, 1:] - x_train_cpu[:, :, :-1]
        x_test_diff_cpu[:, :, 1:] = x_test_cpu[:, :, 1:] - x_test_cpu[:, :, :-1]
        x_train_diff = torch.tensor(x_train_diff_cpu).float().cuda()
        x_test_diff = torch.tensor(x_test_diff_cpu).float().cuda()
        if set == 'Train':
            target_sample = x_train_diff[id:id + 1]
            target_sample_in_time = x_train[id:id + 1]
        else:
            target_sample = x_test_diff[id:id + 1]
            target_sample_in_time = x_test[id:id + 1]

        predicted_label = torch.argmax(explain_model(target_sample)).item()
        exp = explain(xai, xai_name, target_sample, predicted_label, sliding_window=(1, 5), baselines=x_train[-40:])

        # We ignore the first step because it stores the starting value. Note that the first value is not a difference.
        # More importantly, because it shows the starting point, similar to min value in min zero space, it affects all time steps.
        exp[:, :, 0] = 0

        classifier_robustness, xai_robustness = Robustness(target_sample, exp, explain_model, xai, xai_name, x_space, sliding_window=(1, 5), baselines=x_train[-40:])
        paf = PAF(target_sample, exp, explain_model, predicted_label, type=x_space,
                  epsilon=0.01)  # -1 means unsuccessful explanation

        exp = exp.detach().cpu().clone().numpy()
        if x_space == 'Diff_back_to_Time':
            plot_attribution(np.array([target_sample_in_time.cpu().numpy()]), exp, figsize=(12, 6), title="Difference mapped back to Time domain")
        else:
            plot_attribution(np.array([target_sample.cpu().numpy()]), exp, figsize=(12, 6), title="Difference Space")


    print("Classifier Robustness: ", classifier_robustness)
    print("XAI Robustness: ", xai_robustness)
    print("Percentage of Attribution to Flip the label (-1 means zeroing all features with non-zero attribution doesn't flip the label): ", paf)
    print("Sparsity: ", sparsity(exp))

