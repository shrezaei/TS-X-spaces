import torch
import torchaudio
import torch.nn as nn


class FFT_Wrapper(nn.Module):
    def __init__(self, base_model, base_model_input_length):
        super().__init__()
        self.base_model = base_model
        self.l_ifft = torch.fft.irfft
        self.ts_length = base_model_input_length

    def forward(self, x):
        z_amp = torch.exp(x[:, 0]) - 1
        z_phs = x[:, 1]
        z = z_amp * torch.exp(1j * z_phs)
        z = torch.real(self.l_ifft(z, n=self.ts_length, dim=1))
        z = z.view(z.shape[0], 1, -1)
        z = self.base_model(z)
        return z


class Spectrogram_Wrapper(nn.Module):
    def __init__(self, base_model, n_fft=100):
        super().__init__()
        self.base_model = base_model
        self.n_fft = n_fft
        self.l_ispec = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=int(n_fft/2)).cuda()

    def forward(self, x):
        z_amp = x[:, 0]
        z_phs = x[:, 1]
        z = z_amp * torch.exp(1j * z_phs)
        z = torch.real(self.l_ispec(z))
        z = z.view(z.shape[0], 1, -1)
        z = self.base_model(z)
        return z


class Diff_Wrapper(nn.Module):
    def __init__(self, base_model, input_size):
        super().__init__()
        self.base_model = base_model
        self.undiff = nn.Linear(input_size, input_size, bias=False)
        w_matrix = torch.autograd.Variable(torch.ones(input_size, input_size), requires_grad=True).float().cuda()
        w_matrix = torch.tril(w_matrix)
        self.undiff.weight.data = w_matrix

    def forward(self, x):
        x = self.undiff(x)
        x = self.base_model(x)
        return x


class Min_Zero_Wrapper(nn.Module):
    def __init__(self, base_model, input_size):
        super().__init__()
        self.base_model = base_model
        self.mean_zerod = nn.Linear(input_size+1, input_size, bias=False)
        w_matrix = torch.concatenate((torch.diag(torch.ones(input_size)), +1 * torch.ones(input_size, 1)), dim=1).float().cuda()
        self.mean_zerod.weight.data = w_matrix

    def forward(self, x):
        x = self.mean_zerod(x)
        x = self.base_model(x)
        return x
