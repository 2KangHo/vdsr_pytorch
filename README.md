# VDSR PyTorch
[VDSR](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf) PyTorch Implementation  
You can use multi-gpus.  
but no multi-scale.  
And you can input gaussian noise to input images.

## Requirement
`torch`  
`torchvision`  
`python-tk` (or `python3-tk`)

## Download dataset
1. Download [DF2K dataset](https://drive.google.com/file/d/1P9pcaGjvq3xiF22GXIq7ciZta3rjZxaY/view?usp=sharing). Or other dataset is ok, but directory hierarchy -> `<NAME>/train/`, `<NAME>/valid/`
2. move under dataset directory (i.e. `vdsr_pytorch/dataset/DF2K`)

## Usage
### Training & Evaluation
```
usage: main.py [-h] --dataset DATASET --crop_size CROP_SIZE
               --upscale_factor UPSCALE_FACTOR [--batch_size BATCH_SIZE]
               [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--step STEP] [--clip CLIP] [--weight-decay WEIGHT_DECAY]
               [--cuda] [--threads THREADS] [--gpuids GPUIDS [GPUIDS ...]]
               [--add_noise] [--noise_std NOISE_STD] [--test] [--model PATH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset directory name
  --crop_size CROP_SIZE
                        network input size
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
  --batch_size BATCH_SIZE
                        training batch size
  --test_batch_size TEST_BATCH_SIZE
                        testing batch size
  --epochs EPOCHS       number of epochs to train for
  --lr LR               Learning Rate. Default=0.001
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --clip CLIP           Clipping Gradients. Default=0.4
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        Weight decay, Default: 1e-4
  --cuda                use cuda?
  --threads THREADS     number of threads for data loader to use
  --gpuids GPUIDS [GPUIDS ...]
                        GPU ID for using
  --add_noise           add gaussian noise?
  --noise_std NOISE_STD
                        standard deviation of gaussian noise
  --test                test mode
  --model PATH          path to test or resume model
```

#### Example for training
```
> python main.py --dataset DF2K --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --batch_size 128 --test_batch_size 32 --epochs 100
```
or
```
> python3 main.py --dataset DF2K --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --batch_size 128 --test_batch_size 32 --epochs 100
```

#### Example for evaluation
```
> python main.py --dataset Urban100 --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --test_batch_size 32 --test --model model_epoch_100.pth
```
or
```
> python3 main.py --dataset Urban100 --cuda --gpuids 0 1 --upscale_factor 2 --crop_size 256 --test_batch_size 32 --test --model model_epoch_100.pth
```

### Sample usage
```
usage: run.py [-h] --input_image INPUT_IMAGE --model MODEL
              [--output_filename OUTPUT_FILENAME]
              [--scale_factor SCALE_FACTOR] [--cuda]
              [--gpuids GPUIDS [GPUIDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --input_image INPUT_IMAGE
                        input image to use
  --model MODEL         model file to use
  --output_filename OUTPUT_FILENAME
                        where to save the output image
  --scale_factor SCALE_FACTOR
                        factor by which super resolution needed
  --cuda                use cuda
  --gpuids GPUIDS [GPUIDS ...]
                        GPU ID for using
```

#### Example for demonstration
```
> python run.py --cuda --gpuids 0 1 --scale_factor 2 --model model_epoch_100.pth --input_image test_scale2x.jpg --output_filename test_scale2x_out.jpg
```
or
```
> python3 run.py --cuda --gpuids 0 1 --scale_factor 2 --model model_epoch_100.pth --input_image test_scale2x.jpg --output_filename test_scale2x_out.jpg
```

