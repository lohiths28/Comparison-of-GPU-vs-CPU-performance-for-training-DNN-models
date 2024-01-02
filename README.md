
# Comparison of GPU vs CPU Perfomance for training DNN Models

Tensors are a type of data structure used in linear algebra, and like vectors and matrices.Tensors because of its inherent nature are used as default data structure in popular Deep learning Packages like Tensorflow and Pytorch.

## Setting up GPU environment

run this command to check status of drivers,
```bash
nvidia-smi
```
check the drivers in from the list
 
to manually install drivers
```bash
sudo add-apt-repository ppa:graphics-driver/ppa
```
Create an virtual environment using virtualenv

```bash
pip install virtualenv
virtualenv gpu
source gpu/bin/activate
```
Download CUDA toolkit from the site:
https://developer.nvidia.com/cuda-downloads

For ubuntu use:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
```
install the download

```bash
sudo dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-debian11-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```
NOTE : Make sure to install compatible version for the installed drivers

Install CudaDNN: https://developer.nvidia.com/cudnn

Install GPU version of Pytorch and Tensorflow

Pytorch : https://pytorch.org/get-started/locally/

Tensorflow : https://www.tensorflow.org/install

Set device as GPU in the respective code.

## 🔗 Links
CPU VS GPU COMPARISON IN TENSORFLOW: 
https://datamadness.github.io/TensorFlow2-CPU-vs-GPU

VIDEO LINK :
https://drive.google.com/file/d/1yZusxwT8kzJ9vC_f-hfdM_L2zejpfoJ4/view?usp=sharing
