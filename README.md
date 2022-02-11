# Description
In this repository, i summarzie and organize how to use gpu for training and implementing DNN by using CUDA, cuDNN.  

## CUDA, cuDNN  
CUDA is not programming language like C. It is an API model for being used to use GPU resource. In other words it is **the interface** between CPU and GPU. By using api function in CUDA, we can use GPU. cuDNN support DNN implementation on gpu in CUDA. In pytorch, we can use gpu for training and implement DNN by using CUDA and cuDNN, but do not need to directly use api function in cuDNN. By using pytorch's api function which support to use CUDA and cuDNN, we can easily use gpu. 

## Set up guide
I refer to a site: https://velog.io/@xdfc1745/CUDA-CuDNN-%EC%84%A4%EC%B9%98 for setting up.  

## How to use CUDA, cuDNN in torch  
In pytorch docs, there are "torch.backends" and "torch.cuda" class.   
### torch.backends 
torch.backends controls the behavior of various backends that PyTorch supports. In that class, there are "torch.backends.cuda", "torch.backends.cudnn". 
* torch.backends.cuda  

* torch.backends.cudnn  

### torch.cuda   
This package adds support for CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation.  


# Reference  
https://velog.io/@xdfc1745/CUDA-CuDNN-%EC%84%A4%EC%B9%98

