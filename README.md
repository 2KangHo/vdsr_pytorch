# vdsr_pytorch_lms
VDSR PyTorch Implementation  
You can use multi-gpus.  
but no multi-scale.  
And you can input gaussian noise to input images.

## Requirement
`torch`  
`torchvision`  
`python-tk` (or `python3-tk`)

## Download dataset
1. Download [DF2K dataset](https://drive.google.com/file/d/1P9pcaGjvq3xiF22GXIq7ciZta3rjZxaY/view?usp=sharing).
2. move under dataset directory i.e. vdsr_pytorch_lms/dataset/DF2K

## Training
```
$ python main.py --dataset DF2K --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --batch_size 128 --test_batch_size 32
```
or
```
$ python3 main.py --dataset DF2K --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --batch_size 128 --test_batch_size 32
```

## Test
```
$ python main.py --dataset Urban100 --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --test_batch_size 32 --test --model model_epoch_100.pth
```
or
```
$ python3 main.py --dataset Urban100 --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --test_batch_size 32 --test --model model_epoch_100.pth
```

## Sample usage
```
$ python run.py --cuda --gpuids 0 1 --scale_factor 2 --model model_epoch_100.pth --input_image test_scale2x.jpg --output_filename test_scale2x_out.jpg
```
or
```
$ python3 run.py --cuda --gpuids 0 1 --scale_factor 2 --model model_epoch_100.pth --input_image test_scale2x.jpg --output_filename test_scale2x_out.jpg
```

