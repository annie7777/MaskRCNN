# Ubuntu 18.04 CUDA 10.0 
-----------------------------------------------------------------------------------------------------------------------------
# MaskRCNN Installation 

## Step 1: Install Anaconda3 from

https://docs.anaconda.com/anaconda/install/linux/

also don't forget

    $ export PATH=/home/mtrn-server02/anaconda3/bin:$PATH^C

## Step2: Create a conda virtual environment

#we will be using Anaconda with python 3.6.

#run this command in terminal

    $ conda create -n MaskRCNN python=3.6 pip

## Step3: Clone the MaskRCNN repo

    $ cd ~/Documents
    $ mkdir Mask
    $ cd Mask
    $ git clone https://github.com/matterport/Mask_RCNN.git

## Step4: Install the Dependencies¶
    $ source activate MaskRCNN
    $ which pip
    $ which python
    See if both are for conda evns Python 3.6
    $ pip install -r requirements.txt (if "GPU" remove tesnorflow, tensorflow will need addtional installation, see Session2, if "CPU", you don't need to reomove and no need to see Session2, but remember changing "works=1, multiprocessing=False")

## Step5: Install MaskRCNN

    $ python3 setup.py install

## Step6: Install pycocotools¶

    $ git clone https://github.com/waleedka/coco
    $ cd ~/coco/PythonAPI
    $ make
    
Dont't forget to put pycoco folder in the MaskRCNN-master

## Troubleshooting:

1. If you get stuck in random steps:

Go to ~MaskRCNN-master/mrcnn/model.py, change "works=1, multiprocessing=False"

2. If not able to import modules

Check whether you install them after source activate MaskRCNN or using the correct pip
pip should be inside the conda virtual envs

----------------------------------------------------------------------------------------------------------------------------
# Tensorflow Installation

#My machine is Cuda 10.0-only GPU so I installed Cuda 10.0 with tensorflow-gpu-1.12. You cannot intall tensorflow-gpu-1.12 by using pip as system pip is not compatiable to CUDA 10.0,for it was build by CUDA 9.0,so if you want to use the latest version tensorflow-gpu with CUDA 10.0 in 18.04,you need to build from source.This is going to be a tutorial on how to install tensorflow 1.12 GPU version. We will also be installing CUDA 10.0 and cuDNN 7.3.1 along with the GPU version of tensorflow 1.12.

## Step 1: Update and Upgrade your system

#suggest to change the apt source to local sites

    $ sudo apt-get update
    $ sudo apt-get upgrade

## Step 2: Verify You Have a CUDA-Capable GPU

    $ lspci | grep -i nvidia

#Note GPU model. eg. RTX2070

## Step 3: Verify You Have a Supported Version of Linux

#To determine which distribution and release number you’re running, type the following at the command line:

    $ uname -m && cat /etc/*release

#The x86_64 line indicates you are running on a 64-bit system which is supported by cuda 10

## Step 4: Install Dependencies

#Required to compile from source:

    $ sudo apt-get install build-essential
    $ sudo apt-get install cmake git unzip zip

## Step 5: Install linux kernel header

#Goto terminal and type:

    $ uname -r

#You can get like “4.15.0-36-generic”. Note down linux kernel version.
#To install linux header supported by your linux kernel do following:

    $ sudo apt-get install linux-headers-$(uname -r)

## Step 6: Install NVIDIA CUDA 10.0

#If you have been installed any version of CUDA,remove previous cuda installation use follow action:

    $ sudo apt-get purge nvidia*
    $ sudo apt-get autoremove
    $ sudo apt-get autoclean
    $ sudo rm -rf /usr/local/cuda*

#Install cuda For Ubuntu 18.04 :

    $ sudo apt install dracut-core
    $ sudo gedit /etc/modprobe.d/blacklist.conf
#append one line to the file: blacklist nouveau
   
    $ sudo update-initramfs -u
    $ reboot

#Download CUDA install files,download address:CUDA Toolkit 10.0 Download.
#Go to your download folders

    $ sudo sh cuda_10.0.130_410.48_linux.run

## Step 7: Reboot the system to load the NVIDIA drivers.

    $ reboot

## Step 8: Go to terminal and type

    $ echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
    $ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    $ source ~/.bashrc
    $ sudo ldconfig
    $ nvidia-smi
#Check driver version probably Driver Version: 410.48

## Step 9: Verifying your cuda

#You can check your cuda installation using following sample:

    $ cd ~/NVIDIA_CUDA-10.0_Samples/1_Utilities/deviceQuery
    $ make
    $ ./deviceQuery
#the result will show the information about you GPU devices

## Step 10: Install cuDNN 7.3.1

#NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.
#Goto NVIDIA cuDNN and download Login and agreement required,You can download the file without login in the follow address:
#Download the following:
#cuDNN v7.3.1 Library for Linux [ cuda 10.0]

#After downloaded folder and in terminal perform following:

    $ tar -xf cudnn-10.0-linux-x64-v7.3.1.20.tgz
    $ sudo cp -R cuda/include/* /usr/local/cuda-10.0/include
    $ sudo cp -R cuda/lib64/* /usr/local/cuda-10.0/lib64

## Step 11: Install NCCL 2.3.7

#NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives #that are performance optimized for NVIDIA GPUs

#Go and attend survey to https://developer.nvidia.com/nccl/nccl-download to download Nvidia NCCL.
#Download following:
#Donload NCCL v2.3.7, for CUDA 10.0 -> NCCL 2.3.5 O/S agnostic and CUDA 10.0(x86)
#Goto downloaded folder and in terminal perform following:

    $ tar -xf nccl_2.3.5-2+cuda10.0_x86_64.txz
    $ cd nccl_2.3.5-2+cuda10.0_x86_64
    $ cd /usr/local/cuda-10.0/
    $ mkdir targets
    $ cd targets
    $ mkdir x86_64-linux
    $ sudo cp -R * /usr/local/cuda-10.0/targets/x86_64-linux/
    $ sudo ldconfig

## Step 12: Installing Bazel

#Download bazel to your virtual environment created in Section 1
    
    $ cd /home/anaconda3/envs/MaskRCNN/lib/python3.6/site_packages
    $ mkdir bazel && cd bazel
    $ wget https://github.com/bazelbuild/bazel/releases/download/0.12.0/bazel-0.12.0-dist.zip

    $ unzip bazel-0.12.0-dist.zip
    $ bash ./compile.sh
    $ export PATH=`pwd`/output:$PATH

## Step 13: Installing TensorFlow-GPU

    $ souce activate MaskRCNN
    $ cd /home/anaconda3/envs/MaskRCNN/lib/python3.6/site_packages
    $ git clone https://github.com/tensorflow/tensorflow
    $ cd tensorflow
    $ git checkout r1.12

    $ ./configure
==============================================TERMINAL======================================================
Give python path in
Please specify the location of python. [Default is .../anaconda3/envs/MaskRCNN/lib/python3.6]
Press enter two times

Do you wish to build TensorFlow with Apache Ignite support? [Y/n]: Y
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: Y
Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
Do you wish to build TensorFlow with ROCm support? [y/N]: N
Do you wish to build TensorFlow with CUDA support? [y/N]: Y

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.0

Please specify the location where CUDA 10.0 toolkit is installed. Refer to Home for more details. [Default is /usr/local/cuda]: /usr/local/cuda-10.0

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7.3.1

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda-10.0/

Do you wish to build TensorFlow with TensorRT support? [y/N]: N

Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 2.3.7

Please specify the location where NCCL 2.3.5 is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda-10.0/targets/x86_64-linux/

Now we need compute capability which we have noted at step 1 eg. 5.0

Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 7.5] 7.5

Do you want to use clang as CUDA compiler? [y/N]: N

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc

Do you wish to build TensorFlow with MPI support? [y/N]: N

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=native

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:N

Configuration finished

==============================================================================================================
## Step 13: Build Tensorflow using bazel
#The next step in the process to install tensorflow GPU version will be to build tensorflow using bazel. This process takes #a fairly long time.

#To build a pip package for TensorFlow you would typically invoke the following command:

    $ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

#The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the tensorflow_pkg directory:
#To build whl file issue following command:

    $ bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg

#To install tensorflow with pip:
    
    $ cd tensorflow_pkg
    $ sudo pip install tensorflow*.whl

#Note : if you got error like unsupported platform then make sure you are running correct pip command associated with the #python you used while configuring tensorflow build.

## Step 14: Verify Tensorflow installation

    $ python
    $ import tensorflow as tf
    $ hello = tf.constant('Hello, TensorFlow!')
    $ sess = tf.Session()
    $ print(sess.run(hello))

The system outputs Tensorflow load information and the 'Hello,TensorFlow!'
Success! You have now successfully installed tensorflow-gpu 1.12 on your machine.

# Troubleshooting 
after install CuDNN if got the problem "RuntimeError: CUDNN_STATUS_INTERNAL_ERROR #32
"  Just sudo rm -rf ~/.nv and reboot, everything will be ok!(See: https://github.com/SeanNaren/deepspeech.pytorch/issues/32)

Ref: 
https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03
https://zhuanlan.zhihu.com/p/49166211

----------------------------------------------------------------------------------------------------------------------------
useful commands:

htop

nvidia-smi
