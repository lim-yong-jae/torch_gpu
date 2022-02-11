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
print(torch.cuda.current_device()) # get device index
print(torch.cuda.device_count()) # check available gpu device.

if torch.cuda.is_available() == True:
    device = torch.cuda.current_device() # get device index
    print(torch.cuda.get_device_name(device)) # get device name.
    torch.cuda.set_device(device) # set current device.
```   

## Create model on gpu
```python  
import torch
import model

# designate device
device = torch.device('cpu')
if torch.cuda.is_available() == True:
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    device = torch.device('cuda', device)

# Create model
hidden = [100, 50, 25]
net = model.Actor(10, -2, 2, hidden).to(device)

# check model is on cuda
print(next(net.parameters()).is_cuda)
```  

## save and load model on gpu
```python  
# cuda device
device = torch.device('cuda', torch.cuda.current_device())

# save
PATH = "model.pt"
torch.save(net.state_dict(), PATH) # Save model on GPU, CPU

# Load
load_net = model.Actor(10, -2, 2, hidden)
load_net.load_state_dict(torch.load(PATH))
load_net.to(device) # load on CPU
print(next(load_net.parameters()).is_cuda)
```  

## model train using gpu
If DNN is on gpu, DNN's input data should on gpu too. For calculating loss, True answer should be on gpu too, because loaded tensor on gpu should be calculated with tensor which is loaded on gpu.  

```python  
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # Tensor shoule be on gpu, becasue DNN is on gpu. 

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Releases all unoccupied cached memory currently held by
# the caching allocator so that those can be used in other
# GPU application and visible in nvidia-smi
torch.cuda.empty_cache() 
```  

# Reference  
* torch, cuda, cudnn install guide: https://velog.io/@xdfc1745/CUDA-CuDNN-%EC%84%A4%EC%B9%98
* implementation guide: https://velog.io/@papago2355/Pytorch%EC%97%90%EC%84%9C-GPU%EB%A5%BC-%EC%93%B0%EA%B3%A0-%EC%8B%B6%EC%96%B4%EC%9A%94%EC%84%A4%EC%B9%98%EB%A5%BC-%EB%81%9D%EB%82%B8%EB%92%A4%EC%97%90-%EC%BD%94%EB%93%9C%EC%97%90%EC%84%9C-%EB%82%B4%EA%B0%80-%ED%95%B4%EC%95%BC%ED%95%A0%EA%B2%83
* pytorch make model and train tutorial: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* pytorch gpu tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
* Tip: https://yonghyuc.wordpress.com/2019/08/06/pytorch-cuda-gpu-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/
