# Description
In this repository, i summarzie and organize how to use gpu for training and implementing DNN by using CUDA, cuDNN.  

## CUDA, cuDNN  
CUDA is not programming language like C. It is an API model for being used to use GPU resource. In other words it is **the interface** between CPU and GPU. By using api function in CUDA, we can use GPU. cuDNN support DNN implementation on gpu in CUDA. In pytorch, we can use gpu for training and implement DNN by using CUDA and cuDNN, but do not need to directly use api function in cuDNN. By using pytorch's api function which support to use CUDA and cuDNN, we can easily use gpu. 

## Set up guide
I refer to a site: https://velog.io/@xdfc1745/CUDA-CuDNN-%EC%84%A4%EC%B9%98 for setting up.  

## How to use CUDA, cuDNN in torch  
In pytorch docs, there are "torch.backends" and "torch.cuda" class.   
### torch.backends 
torch.backends controls the behavior of various backends that PyTorch supports. In that class, there are "torch.backends.cuda", "torch.backends.cudnn" and so on. 
* torch.backends.cuda  
  By using that class, we can check torch is built on cuda, or control whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs, and so on. 
* torch.backends.cudnn  
  By using that class, we can check cuDNN version and, control whether cuDNN is enabled, and so on.  
  
### torch.cuda   
This package adds support for CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation.   

```python  
import torch

print(torch.cuda.is_available()) # check torch is built on cuda.
print(torch.cuda.current_device()) # check current connected gpu device.
print(torch.cuda.device_count()) # check available gpu device.

device = "cpu"

if torch.cuda.is_available() == True:
    device = torch.cuda.current_device()
    print(torch.cuda.get_device_name(device)) # get device name.

torch.cuda.device(device) # select device.
```   


## Create model on gpu

## save and load model on gpu

## model train using gpu


# Reference  
* torch, cuda, cudnn install guide: https://velog.io/@xdfc1745/CUDA-CuDNN-%EC%84%A4%EC%B9%98
* implementation guide: https://velog.io/@papago2355/Pytorch%EC%97%90%EC%84%9C-GPU%EB%A5%BC-%EC%93%B0%EA%B3%A0-%EC%8B%B6%EC%96%B4%EC%9A%94%EC%84%A4%EC%B9%98%EB%A5%BC-%EB%81%9D%EB%82%B8%EB%92%A4%EC%97%90-%EC%BD%94%EB%93%9C%EC%97%90%EC%84%9C-%EB%82%B4%EA%B0%80-%ED%95%B4%EC%95%BC%ED%95%A0%EA%B2%83
* pytorch make model and train tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* pytorch gpu tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
