# Installing Tensorflow-GPU with NVIDIA CUDA 9.0 on a Google Cloud Platform VM instance Ubuntu 16.04
-----------------------------------------------------------------------------------------------------------------------------
# Step 1: Setup a google cloud vm instance
## Create a vm instance
![GitHub Logo](/images/googlevm_instance.PNG)
## Click on Management, disks, networking, SSH keys option. Add the startup script for ubuntu-16.04 LTS and CUDA-9.
This will automatically download the NVIDIA GPU drivers and CUDA
![GitHub Logo](/images/startup_scipt.PNG)

The latest version startup script can be found [here](https://cloud.google.com/compute/docs/gpus/add-gpus) but I am using the old version for ubuntu-16.04 LTS and CUDA-9.
```
#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  apt-get update
  apt-get install cuda-9-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1
```
## Troubleshooting: 
1. Startup script did not install automatically.

For me, I got an error 'Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)'

Following this [post](https://askubuntu.com/questions/1109982/e-could-not-get-lock-var-lib-dpkg-lock-frontend-open-11-resource-temporari), I solved my problem.

After that, run the command below ([Ref](https://github.com/GoogleCloudPlatform/compute-image-packages/issues/342)).
```
sudo google_metadata_script_runner --script-type=startup --debug
```

Finally, test if everthing is working by running
```
nvidia-smi
nvcc --version
```
which will give you details about your GPU(ps: cuda here is 10.0 not 9.0, check the post [here](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi). The simple answer is that: nvidia-smi shows you the CUDA version that your driver supports. You have one of the recent 410.x drivers installed which support CUDA 10. The version the driver supports has nothing to do with the version you compile and link your program against. A driver that supports CUDA 10.0 will also be able to run an application that was built for CUDA 9.2)
![GitHub Logo](/images/nvdia.PNG)
![GitHub Logo](/images/nvcc.PNG)
# Step 2: Install cudnn
```
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.3.1.20.tgz
sudo tar -xzvf cudnn-9.0-linux-x64-v7.3.1.20.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

Add these lines to end of ~/.bashrc:
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="$PATH:/usr/local/cuda/bin
```
Reload bashrc
```
source ~/.bashrc
```
# Step 3: Install [Anaconda](https://www.anaconda.com/distribution/#linux)
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
chmod +x Anaconda3-2019.03-Linux-x86_64.sh
sudo ./Anaconda3-2019.03-Linux-x86_64.sh
export PATH=/home/mtrn-server02/anaconda3/bin:$PATH^C
```
# Step 4: Install Tensorflow-gpu
## Create a conda virtual environment
```
conda create -n FCN python=3.6
conda activate FCN
```
## Install tensorflow
```
pip install tensorflow-gpu
```
## Test tf install
```
# start python in terminal
python
```
```
import tensorflow as tf
print(tf.__version__)
```
## Troubleshooting:
For me, I got an error 'Failed to load the native TensorFlow runtime : error while importing tensorflow'
I reinstalled tensorflow-gpu with [conda](https://anaconda.org/anaconda/tensorflow-gpu) and it worked. 
Ref is [here](https://github.com/tensorflow/tensorflow/issues/10026).
```
conda install -c anaconda tensorflow-gpu 

# then export path again
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="$PATH:/usr/local/cuda/bin

```

# Ref: 

Installation instruction:

1. https://medium.com/@jayden.chua/quick-install-cuda-on-google-cloud-compute-6c85447f86a1

2. https://medium.com/searce/installing-tensorflow-gpu-with-nvidia-cuda-on-a-google-cloud-platform-vm-instance-b059ea47e55c

3. https://github.com/williamFalcon/tensorflow-gpu-install-ubuntu-16.04/tree/1bf17959b4a757889b774a4fbbd5656904a0afa7

NVIDIA error:

1. https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47252

2. https://towardsdatascience.com/troubleshooting-gcp-cuda-nvidia-docker-and-keeping-it-running-d5c8b34b6a4c

Transfering files to Instances: 

1. https://cloud.google.com/compute/docs/instances/transfer-files



