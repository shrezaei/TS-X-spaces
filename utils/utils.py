import os
import numpy as np
import torch.nn as nn
import torch
import math


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def evaluate_classifier(model, test_loader):
    model.eval()
    correct = 0
    for x_batch, y_batch in test_loader:
        z = model(x_batch)
        y_pred = torch.argmax(z.data, 1)
        y_lbl = torch.argmax(y_batch, 1)
        correct += (y_pred == y_lbl).sum()
    acc = correct / len(test_loader.dataset)
    return acc


def sparsity(a):
    a_abs = np.abs(a)
    a_abs = a_abs / np.max(a_abs)
    a_abs[a_abs < 0.01] = 0
    a_abs = a_abs.reshape(-1)
    return math.pow(np.sum(1-a_abs) / len(a_abs-1), 4)


# Percentage of Attribution to Flip the label (PAF)
def PAF(input, explanation, classifier, predicted_label, type, epsilon=0.0):
    # softMax = nn.Softmax(dim=1)

    assert explanation.shape[0] == 1, "PAF function only accepts one input, not a batch!"
    assert input.shape[0] == 1, "PAF function only accepts one input, not a batch!"
    inp = torch.clone(input)
    exp = torch.clone(explanation)

    # Note that we ignore negative attributions because they are supposed to be in favor of the other classes.
    # Hence, we do not consider them when it comes to faithfulness.
    # They are not supposed to flip the prediction in favor of other classes

    if type == "Freq":
        exp = exp[0, 0, :]      # ignoring the phase content
        exp[exp < epsilon * torch.max(exp).data] = 0
        num_of_attributions = torch.sum(exp > 0.0)
        # print(num_of_attributions, exp.shape, torch.max(exp).data)
        _, indices = torch.sort(exp, descending=True)
        for i in range(0, num_of_attributions):
            # print(indices[i])
            inp[0, :, indices[i]] = 0
            new_input_label = torch.argmax(classifier(inp)).item()
            if predicted_label != new_input_label:
                # print(predicted_label, new_input_label)
                return ((i+1) / num_of_attributions).cpu().detach().numpy() * 100.0
        return -1

    elif type in ["Time", "meanz"]:
        assert input.shape[1] == 1, "The package doesn't support multivariate time series!"
        exp = exp[0, 0, :]
        exp[exp < epsilon * torch.max(exp).data] = 0
        num_of_attributions = torch.sum(exp > 0.0)
        # print(num_of_attributions, exp.shape, torch.max(exp).data)
        _, indices = torch.sort(exp, descending=True)
        for i in range(0, num_of_attributions):
            # print(indices[i])
            inp[0, :, indices[i]] = 0
            new_input_label = torch.argmax(classifier(inp)).item()
            if predicted_label != new_input_label:
                # print(predicted_label, new_input_label)
                return ((i+1) / num_of_attributions).cpu().detach().numpy() * 100.0
        return -1

    elif type in ["Diff", "Diff_back_to_Time"]:
        assert input.shape[1] == 1, "The package doesn't support multivariate time series!"
        exp = exp[0, 0, :]
        exp[exp < epsilon * torch.max(exp).data] = 0
        exp[0] = 0    # The element stores the min value which should be ignored for explanation purposes
        num_of_attributions = torch.sum(exp > 0.0)
        # print(num_of_attributions, exp.shape, torch.max(exp).data)
        _, indices = torch.sort(exp, descending=True)
        for i in range(0, num_of_attributions):
            # print(indices[i])
            inp[0, :, indices[i]] = 0
            new_input_label = torch.argmax(classifier(inp)).item()
            # print(indices[i], predicted_label, new_input_label)
            if predicted_label != new_input_label:
                # print(predicted_label, new_input_label)
                return ((i + 1) / num_of_attributions).cpu().detach().numpy() * 100.0
        return -1
    elif type in ["MinZero"]:
        assert input.shape[1] == 1, "The package doesn't support multivariate time series!"
        exp = exp[0, 0, :]
        exp[exp < epsilon * torch.max(exp).data] = 0
        exp[-1] = 0    # The element stores the min value which should be ignored for explanation purposes
        num_of_attributions = torch.sum(exp > 0.0)
        # print(num_of_attributions, exp.shape, torch.max(exp).data)
        _, indices = torch.sort(exp, descending=True)
        for i in range(num_of_attributions):
            # print(indices[i])
            inp[0, :, indices[i]] = 0
            new_input_label = torch.argmax(classifier(inp)).item()
            # print(indices[i], predicted_label, new_input_label)
            if predicted_label != new_input_label:
                # print(predicted_label, new_input_label)
                return ((i + 1) / num_of_attributions).cpu().detach().numpy() * 100.0
        return -1
    elif type == "TimeFreq":
        input_size = inp.shape
        exp_flattened = torch.reshape(exp, (1, 2, -1))
        input_flattened = torch.reshape(input, (1, 2, -1))
        exp_flattened[:, 1, :] = 0  # No attribution is considered for phase information
        exp_flattened[0, 0, exp_flattened[0, 0, :] < epsilon * torch.max(exp_flattened).data] = 0
        num_of_attributions = torch.sum(exp > 0.0)
        _, indices = torch.sort(exp_flattened, descending=True)
        for i in range(num_of_attributions):
            index = indices[0, 0, i]
            input_flattened[0, :, index] = 0
            input_rewinded = torch.reshape(input_flattened, input_size)
            new_input_label = torch.argmax(classifier(input_rewinded)).item()
            if predicted_label != new_input_label:
                # print(predicted_label, new_input_label)
                return ((i + 1) / num_of_attributions).cpu().detach().numpy() * 100.0
        return -1

    return -1



def Robustness(input, exp, classifier, xai_method, xai_name, type, sliding_window=(1, 10), baselines=None, num_perturbations=10, epsilon=0.01):
    softMax = nn.Softmax(dim=1)
    features = input.shape[1]
    steps = input.shape[2]

    perturbed_input = torch.vstack([input] * num_perturbations)
    # print(perturbed_input.shape)

    # std_avg = 0
    for i in range(features):
        std = torch.std(input[0, i, :])
        # std_avg += std
        noise = torch.normal(0, std * epsilon * torch.ones(size=(num_perturbations, *input.shape[2:]), device="cuda"))
        if type == "Diff_ignored":
            noise[:, 0] = 0
        elif type == "MZ_ignored":
            noise[:, -1] = 0
        perturbed_input[:, i, :] += noise

    # std_avg /= features
    # return 0, 0, std_avg

    input_probs = softMax(classifier(input))
    input_preds = torch.argmax(input_probs[0])
    perturbed_probs = softMax(classifier(perturbed_input))
    # perturbed_preds = torch.argmax(perturbed_probs, dim=1)

    classifier_robustness = np.sum([abs(input_probs[0, input_preds] - perturbed_probs[i, input_preds]).detach().cpu().clone().numpy() for i in range(num_perturbations)]) / num_perturbations
    # print(classifier_robustness)

    perturbed_exp = explain(xai_method, xai_name, perturbed_input, input_preds, sliding_window=sliding_window, baselines=baselines)

    if type == "Diff_ignored":
        exp[:, :, 0] = 0
        perturbed_exp[:, :, 0] = 0
    elif type == "MZ_ignored":
        exp[:, :, -1] = 0
        perturbed_exp[:, :, -1] = 0

    exp = exp.detach().cpu().clone().numpy()
    exp = exp.reshape(1, -1)
    perturbed_exp = perturbed_exp.detach().cpu().clone().numpy()
    perturbed_exp = perturbed_exp.reshape(num_perturbations, -1)

    # Normalize such that the explanation range is [0, 1]
    exp = np.abs(exp) / np.max(np.abs(exp))
    perturbed_exp = np.abs(perturbed_exp)
    for i in range(num_perturbations):
        perturbed_exp[i] = perturbed_exp[i] / np.max(perturbed_exp[i])

    xai_robustness = np.sum([np.linalg.norm(exp-perturbed_exp[i])/exp.shape[1] for i in range(num_perturbations)]) / num_perturbations
    xai_robustness = np.nan_to_num(xai_robustness, nan=-1)
    # print(xai_robustness)

    return classifier_robustness, xai_robustness


def explain(xia_method, xai_name, input, predicted_label, sliding_window=(1, 10), baselines=None):
    if xai_name == "Occlusion":
        exp = xia_method.attribute(input, target=predicted_label, sliding_window_shapes=sliding_window)  # for Occlusion
    elif xai_name == "GradientShap":
        exp = xia_method.attribute(input, baselines=baselines, target=predicted_label)
    elif xai_name in ["Lime", "KernelShap"]:
        exp = torch.zeros(input.shape).float().cuda()
        for i in range(input.shape[0]):
            exp[i] = xia_method.attribute(input[i:i+1], target=predicted_label)
    else:
        exp = xia_method.attribute(input, target=predicted_label)
    return exp

