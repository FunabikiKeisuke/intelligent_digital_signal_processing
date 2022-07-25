import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset.dataset import CelebA, CelebAFake


def image_normalization(src_img):
    """
    白飛び防止のための正規化処理
    cv2.imshowでwavelet変換された画像を表示するときに必要（大きい値を持つ画像の時だけ）
    """
    norm_img = (src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img))
    return norm_img


def merge_images(cA, cH_V_D):
    """numpy.array を４つ(左上、(右上、左下、右下))連結させる"""
    cH, cV, cD = cH_V_D
    cH = image_normalization(cH)  # 外してもok
    cV = image_normalization(cV)  # 外してもok
    cD = image_normalization(cD)  # 外してもok
    cA = cA[0:cH.shape[0], 0:cV.shape[1]]  # 元画像が2の累乗でない場合、端数ができることがあるので、サイズを合わせる。小さい方に合わせます。
    return np.vstack((np.hstack((cA, cH)), np.hstack((cV, cD))))  # 左上、右上、左下、右下、で画素をくっつける


def coeffs_visualization(cof):
    norm_cof0 = cof[0]
    norm_cof0 = image_normalization(norm_cof0)  # 外してもok
    merge = norm_cof0
    for i in range(1, len(cof)):
        merge = merge_images(merge, cof[i])  # ４つの画像を合わせていく
    cv2.imshow('', merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def wavelet_transform_for_image(src_image, level, M_WAVELET="db1", mode="sym"):
    data = src_image.astype(np.float64)
    coeffs = pywt.wavedec2(data, M_WAVELET, level=level, mode=mode)
    return coeffs


if __name__ == "__main__":
    # 'haar', 'db', 'sym' etc...
    # URL: http://pywavelets.readthedocs.io/en/latest/ref/wavelets.html

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
            coeffs = wavelet_transform_for_image(data[:, :, i], level=1, M_WAVELET="haar")
            norm_cof0 = coeffs[0]
            norm_cof0 = image_normalization(norm_cof0)  # 外してもok
            merge = norm_cof0
            for c in range(1, len(coeffs)):
                merge = merge_images(merge, coeffs[c])  # ４つの画像を合わせていく
            data[:, :, i] = merge

        data = np.transpose(data, (2, 0, 1))
        # result_path = f"dataset/celeba_wt/{batch_idx + 1}-images.jpg"
        result_path = f"dataset/celeba_fake_wt/{batch_idx + 1}-images.jpg"
        save_image(torch.from_numpy(data), result_path, nrow=1, padding=0)

    progress_bar = tqdm(test_loader, leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        data = np.squeeze(data.numpy().astype(np.float32)).transpose((1, 2, 0))
        for i in range(3):  # RGB
            coeffs = wavelet_transform_for_image(data[:, :, i], level=1, M_WAVELET="haar")
            norm_cof0 = coeffs[0]
            norm_cof0 = image_normalization(norm_cof0)  # 外してもok
            merge = norm_cof0
            for c in range(1, len(coeffs)):
                merge = merge_images(merge, coeffs[c])  # ４つの画像を合わせていく
            data[:, :, i] = merge

        data = np.transpose(data, (2, 0, 1))
        # result_path = f"dataset/celeba_wt/{batch_idx + 1001}-images.jpg"
        result_path = f"dataset/celeba_fake_wt/{batch_idx + 1001}-images.jpg"
        save_image(torch.from_numpy(data), result_path, nrow=1, padding=0)
