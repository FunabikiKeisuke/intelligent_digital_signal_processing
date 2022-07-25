import zipfile

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


class CelebARealFake(Dataset):
    def __init__(self, train=True, transform=None):
        self.real_dataset_path = "dataset/celeba.zip"
        self.fake_dataset_path = "dataset/celeba_fake.zip"
        self.transform = transform
        self.real_data = []
        self.fake_data = []
        self.real_target = torch.tensor([1.0])
        self.fake_target = torch.tensor([0.0])

        with zipfile.ZipFile(self.real_dataset_path) as zip_images:
            for filepath in zip_images.namelist():
                if filepath[-1] != '/':  # ignore directory path
                    self.real_data.append(filepath)

        with zipfile.ZipFile(self.fake_dataset_path) as zip_images:
            for filepath in zip_images.namelist():
                if filepath[-1] != '/':  # ignore directory path
                    self.fake_data.append(filepath)

        if train:
            self.data = self.real_data[0:1000] + self.fake_data[0:1000]
        else:
            self.data = self.real_data[1000:1200] + self.fake_data[1000:1200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if "fake" not in self.data[idx]:
            with zipfile.ZipFile(self.real_dataset_path) as zip_images:
                image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                return image, self.real_target
        else:
            with zipfile.ZipFile(self.fake_dataset_path) as zip_images:
                image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                return image, self.fake_target


class CelebA(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset_path = "dataset/celeba.zip"
        self.transform = transform
        self.data = []
        self.target = torch.tensor([1.0])

        # with zipfile.ZipFile(self.dataset_path) as zip_images:
        #     for filepath in zip_images.namelist():
        #         if filepath[-1] != '/':  # ignore directory path
        #             self.data.append(filepath)
        for i in range(1, 1201):
            self.data.append(f"celeba/{i}-images.jpg")


        if train:
            self.data = self.data[0:1000]
        else:
            self.data = self.data[1000:1200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.dataset_path) as zip_images:
            image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, self.target


class CelebAFake(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset_path = "dataset/celeba_fake.zip"
        self.transform = transform
        self.data = []
        self.target = torch.tensor([0.0])

        # with zipfile.ZipFile(self.dataset_path) as zip_images:
        #     for filepath in zip_images.namelist():
        #         if filepath[-1] != '/':  # ignore directory path
        #             self.data.append(filepath)
        for i in range(1, 1201):
            self.data.append(f"celeba_fake/{i}-images.jpg")

        if train:
            self.data = self.data[0:1000]
        else:
            self.data = self.data[1000:1200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.dataset_path) as zip_images:
            image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, self.target


class CelebADFTRealFake(Dataset):
    def __init__(self, train=True, transform=None):
        self.real_dataset_path = "dataset/celeba_dft.zip"
        self.fake_dataset_path = "dataset/celeba_fake_dft.zip"
        self.transform = transform
        self.real_data = []
        self.fake_data = []
        self.real_target = torch.tensor([1.0])
        self.fake_target = torch.tensor([0.0])

        with zipfile.ZipFile(self.real_dataset_path) as zip_images:
            for filepath in zip_images.namelist():
                if filepath[-1] != '/':  # ignore directory path
                    self.real_data.append(filepath)

        with zipfile.ZipFile(self.fake_dataset_path) as zip_images:
            for filepath in zip_images.namelist():
                if filepath[-1] != '/':  # ignore directory path
                    self.fake_data.append(filepath)

        if train:
            self.data = self.real_data[0:1000] + self.fake_data[0:1000]
        else:
            self.data = self.real_data[1000:1200] + self.fake_data[1000:1200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if "fake" not in self.data[idx]:
            with zipfile.ZipFile(self.real_dataset_path) as zip_images:
                image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                return image, self.real_target
        else:
            with zipfile.ZipFile(self.fake_dataset_path) as zip_images:
                image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                return image, self.fake_target


class CelebADFT(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset_path = "dataset/celeba_dft.zip"
        self.transform = transform
        self.data = []
        self.target = 1

        with zipfile.ZipFile(self.dataset_path) as zip_images:
            for filepath in zip_images.namelist():
                if filepath[-1] != '/':  # ignore directory path
                    self.data.append(filepath)

        if train:
            self.data = self.data[0:1000]
        else:
            self.data = self.data[1000:1200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.dataset_path) as zip_images:
            image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, self.target


class CelebAFakeDFT(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset_path = "dataset/celeba_fake_dft.zip"
        self.transform = transform
        self.data = []
        self.target = 0

        with zipfile.ZipFile(self.dataset_path) as zip_images:
            for filepath in zip_images.namelist():
                if filepath[-1] != '/':  # ignore directory path
                    self.data.append(filepath)

        if train:
            self.data = self.data[0:1000]
        else:
            self.data = self.data[1000:1200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.dataset_path) as zip_images:
            image = Image.open(zip_images.open(self.data[idx])).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, self.target
