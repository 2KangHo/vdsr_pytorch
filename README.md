# vdsr_pytorch

PyTorch Implementation  
1대1비율의 사진을 2대1비율의 사진으로 만듦.  
progressive영상을 interlaced영상으로 만드는느낌.

## Requirement

`torch`  
`torchvision`  
`python-tk` (or `pyhton3-tk`)

## Training

    > python main.py --batch_size 40 --test_batch_size 10 --epochs 100 --cuda --gpuids 0 --upscale_factor 2

or

    > python3 main.py --batch_size 40 --test_batch_size 10 --epochs 100 --cuda --gpuids 0 --upscale_factor 2

## Test

    > python main.py --batch_size 40 --test_batch_size 10 --epochs 100 --cuda --gpuids 0 --upscale_factor 2 --test --model model_epoch_100.pth

or

    > python3 main.py --batch_size 40 --test_batch_size 10 --epochs 100 --cuda --gpuids 0 --upscale_factor 2 --test --model model_epoch_100.pth

## Sample Usage

    > python run.py --input_image test.jpg --scale_factor 2 --model model_epoch_100.pth --cuda --output_filename <output_filename>

or

    > python3 run.py --input_image test.jpg --scale_factor 2 --model model_epoch_100.pth --cuda --output_filename <output_filename>

## 주의

sample에서 input image는 학습에 사용된 BSDS300 data가 아닌 인터넷에서 가져온 이미지 등 다른 이미지를 사용해야 정확한 성능을 확인할 수 있다.
