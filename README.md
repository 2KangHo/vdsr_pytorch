# vdsr_pytorch

PyTorch Implementation  
no muliti-scale  
but you can use multi-GPUs

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

    > python run.py --input_image test_scale2x.jpg --scale_factor 2 --model model_epoch_100.pth --cuda --gpuids 0 --output_filename test_scale2x_out.jpg

or

    > python3 run.py --input_image test_scale2x.jpg --scale_factor 2 --model model_epoch_100.pth --cuda --gpuids 0 --output_filename test_scale2x_out.jpg

## 주의

test시에는 Urban100의 데이터가 사용된다.  
sample에서 input image는 학습에 사용된 BSDS300 data가 아닌 인터넷에서 가져온 이미지 등 다른 이미지를 사용해야 정확한 성능을 확인할 수 있다.  
Urban100데이터는 학습시 사용하지 않으므로 Urban100의 사진들 중 하나를 사용해도 됨.
