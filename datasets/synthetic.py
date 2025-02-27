import numpy as np


def add_ps(steps, features, add_unfeature=True):
    starting_position = np.random.randint(0, int(steps/2 -1))
    if add_unfeature:
        return features[starting_position:starting_position + steps]
    else:
        return np.zeros(steps)


def add_ts(steps, features, add_unfeature=True, fixed_position=False):
    if fixed_position:
        starting_position = 50
    else:
        starting_position = np.random.randint(0, int(steps - len(features) - 1))
    base = np.zeros(steps)
    base[starting_position:starting_position + len(features)] = features
    if add_unfeature:
        return base
    else:
        return np.zeros(steps)


def freq_features_variable_amplitude_shapelet_feature_per_class(steps=500, train_size=1000, test_size=300, num_classes=2, seed=0, add_freq_unfeature=True, add_shape_unfeature=True, noise_avg=0.0, freq_features=[10, 15], freq_nonfeatures=[33]):
    if seed is not None:
        np.random.seed(seed)

    x_train_cpu = np.random.normal(noise_avg, 0.01, size=(train_size * num_classes, 1, steps))
    y_train_cpu = np.zeros((train_size*num_classes), dtype=int)
    x_test_cpu = np.random.normal(noise_avg, 0.01, size=(test_size * num_classes, 1, steps))
    y_test_cpu = np.zeros((test_size*num_classes), dtype=int)

    ps_feature_c1 = np.sin(np.arange(steps * 2, step=1) * 2 * np.pi * freq_features[0] / steps)
    ps_feature_c2 = np.sin(np.arange(steps * 2, step=1) * 2 * np.pi * freq_features[1] / steps)
    ts_feature_c1 = np.array([0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2])
    ts_feature_c1 = ts_feature_c1 - np.average(ts_feature_c1)
    ts_feature_c2 = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38])
    ts_feature_c2 = ts_feature_c2 - np.average(ts_feature_c2)

    ps_unfeature_1 = np.sin(np.arange(steps * 2, step=1) * 2 * np.pi * freq_nonfeatures[0] / steps)
    ts_unfeature_1 = np.array([-0.1, -0.095, -0.09, -0.08, -0.065, -0.050, -0.030, 0.0, 0.2, 0.5, 0.5, 0.2, 0.0, -0.030, -0.050, -0.065, -0.08, -0.09, -0.095, -0.1])
    ts_unfeature_2 = np.array([0.1, 0.095, 0.09, 0.08, 0.065, 0.050, 0.030, 0.0, -0.2, -0.5, -0.5, -0.2, 0.0, 0.030, 0.050, 0.065, 0.08, 0.09, 0.095, 0.1])

    ps_features = [ps_feature_c1, ps_feature_c2]
    ts_features = [ts_feature_c1, ts_feature_c2]

    for i in range(train_size):
        for j in range(num_classes):
            if i < train_size/2:    #only add ps feature
                amp = np.random.uniform(0.1, 0.2)
                x_train_cpu[num_classes*i+j, 0] += amp * add_ps(steps, ps_features[j])
            else:       #only add ts feature
                ts_mag = np.random.uniform(0.45, 0.55)
                x_train_cpu[num_classes*i+j, 0] += ts_mag * add_ts(steps, ts_features[j])


            if i % 3 == 1 or i % 7 < 2:
                amp = np.random.uniform(0.0, 0.15)
                x_train_cpu[num_classes * i + j, 0] += amp * add_ps(steps, ps_unfeature_1, add_freq_unfeature)
            if i % 3 == 1 or i % 5 == 0:
                x_train_cpu[num_classes * i + j, 0] += add_ts(steps, ts_unfeature_1, add_shape_unfeature)
            if i % 5 == 1 or i % 7 < 2:
                x_train_cpu[num_classes * i + j, 0] += add_ts(steps, ts_unfeature_2, add_shape_unfeature)

    for i in range(test_size):
        for j in range(num_classes):
            # add both ts/ps feature
            amp = np.random.uniform(0.05, 0.25)
            ts_mag = np.random.uniform(0.4, 0.6)
            if num_classes*i+j < 100:
                x_test_cpu[num_classes*i+j, 0] += amp * add_ps(steps, ps_features[j])
                x_test_cpu[num_classes * i + j, 0] += ts_mag * add_ts(steps, ts_features[j])
            if 100 <= num_classes*i+j < 200:
                x_test_cpu[num_classes * i + j, 0] += ts_mag * add_ts(steps, ts_features[j])
            else:
                x_test_cpu[num_classes*i+j, 0] += amp * add_ps(steps, ps_features[j])

            # if add_unfeature:
            if i % 3 == 1 or i % 7 < 2:
                amp = np.random.uniform(0.0, 0.2)
                x_test_cpu[num_classes * i + j, 0] += amp * add_ps(steps, ps_unfeature_1, add_freq_unfeature)
            if i % 3 == 1:
                x_test_cpu[num_classes * i + j, 0] += add_ts(steps, ts_unfeature_1, add_shape_unfeature)
            if i % 5 == 1:
                x_test_cpu[num_classes * i + j, 0] += add_ts(steps, ts_unfeature_2, add_shape_unfeature)

    for j in range(num_classes):
        y_train_cpu[j::num_classes] = j
        y_test_cpu[j::num_classes] = j

    return x_train_cpu, x_test_cpu, y_train_cpu, y_test_cpu


