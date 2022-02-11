# Description
In this repository, i summarzie and organize how to use gpu for training and implementing DNN by using CUDA, cuDNN.  

## CUDA, cuDNN  
CUDA is not programming language like C. It is an API model for being used to use GPU resource. In other words it is **the interface** between CPU and GPU. By using api function in CUDA, we can use GPU. In pytorch, we can use gpu for training and implement DNN by using CUDA and cuDNN, but do not need to directly use api function in CUDA or cuDNN. By using pytorch's api function which support to use CUDA and cuDNN, we can easily use gpu. 

## Set up guide
1) For operating pytorch's DNN on gpu, we need to install CUDA. CUDA version is important, so we should download specific CUDA version which is designated in pytorch.  
<img src="./img/torch.png" alt="MLE" width="80%" height="80%"/>  


2) Download CUDA 
<img src="./img/CUDA.png" alt="MLE" width="80%" height="80%"/> 



# Link  
* torch download link: https://pytorch.org/  
* CUDA(ver 10.2) download link: https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal 
* 
