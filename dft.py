import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset.dataset import CelebA, CelebAFake

transform = transforms.Compose([
    transforms.ToTensor(),
])

# celeba
# train_dataset = CelebA(train=True, transform=transform)
# test_dataset = CelebA(train=False, transform=transform)

# celeba_fake
train_dataset = CelebAFake(train=True, transform=transform)
test_dataset = CelebAFake(train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

progress_bar = tqdm(train_loader, leave=False)
for batch_idx, (data, target) in enumerate(progress_bar):
    data = np.squeeze(data.numpy().astype(np.float32)).transpose((1, 2, 0))
    for i in range(3):  # RGB
        img = data[:, :, i]
        fft_img = np.fft.fft2(img)
        fft_img = np.fft.fftshift(fft_img)
        real = fft_img.real
        real[real > 0] = np.log10(real[real > 0])
        real[real < 0] = -np.log10(-real[real < 0])
        imag = fft_img.imag
        imag[imag > 0] = np.log10(imag[imag > 0])
        imag[imag < 0] = -np.log10(-imag[imag < 0])
        fft_img.real = real
        fft_img.imag = imag
        fft_img = np.abs(fft_img)
        data[:, :, i] = fft_img

    data = np.transpose(data, (2, 0, 1))
    # result_path = f"dataset/celeba_dft/{batch_idx + 1}-images.jpg"
    result_path = f"dataset/celeba_fake_dft/{batch_idx + 1}-images.jpg"
    save_image(torch.from_numpy(data), result_path, nrow=1, padding=0)


progress_bar = tqdm(test_loader, leave=False)
for batch_idx, (data, target) in enumerate(progress_bar):
    data = np.squeeze(data.numpy().astype(np.float32)).transpose((1, 2, 0))
    data = data
    for i in range(3):  # RGB
        img = data[:, :, i]
        fft_img = np.fft.fft2(img)
        fft_img = np.fft.fftshift(fft_img)
        real = fft_img.real
        real[real > 0] = np.log10(real[real > 0])
        real[real < 0] = -np.log10(-real[real < 0])
        imag = fft_img.imag
        imag[imag > 0] = np.log10(imag[imag > 0])
        imag[imag < 0] = -np.log10(-imag[imag < 0])
        fft_img.real = real
        fft_img.imag = imag
        fft_img = np.abs(fft_img)
        data[:, :, i] = fft_img

    data = np.transpose(data, (2, 0, 1))
    # result_path = f"dataset/celeba_dft/{batch_idx + 1001}-images.jpg"
    result_path = f"dataset/celeba_fake_dft/{batch_idx + 1001}-images.jpg"
    save_image(torch.from_numpy(data), result_path, nrow=1, padding=0)
