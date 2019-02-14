from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from data_utils import DatasetFromFolder


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize((crop_size//upscale_factor, crop_size//upscale_factor)),
        Resize((crop_size, crop_size)),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(dataset, crop_size, upscale_factor, add_noise=None, noise_std=3.0):
    root_dir = join("dataset", dataset)
    train_dir = join(root_dir, "train")
    cropsize = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(
                                 cropsize, upscale_factor),
                             target_transform=target_transform(cropsize),
                             add_noise=add_noise,
                             noise_std=noise_std)


def get_validation_set(dataset, crop_size, upscale_factor):
    root_dir = join("dataset", dataset)
    validation_dir = join(root_dir, "valid")
    cropsize = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(validation_dir,
                             input_transform=input_transform(
                                 cropsize, upscale_factor),
                             target_transform=target_transform(cropsize))


def get_test_set(dataset, crop_size, upscale_factor):
    test_dir = join("dataset", dataset)
    cropsize = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(
                                 cropsize, upscale_factor),
                             target_transform=target_transform(cropsize))
