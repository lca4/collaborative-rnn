# Instructions for running the code on AWS EC2.

This document provides step-by-step instructions on running the code on an
Amazon EC2 [p2.xlarge instance][1]. This instance type features an Nvidia K80
GPU.

In EC2's "Launch Instance" wizard, choose the following settings.

- image: Ubuntu Server 16.04 LTS (HVM), SSD Volume Type
- instance type: p2.xlarge
- add storage: increase the the size of the root volume to 30 GiB.

Make sure that you can connect to the instance via SSH from your network
location, by choosing appropriate virtual private cloud and security group
settings. The rest of this guide assumes that you are logged into the instance
via SSH.


## Basic setup

First, we make sure that the operating system is up to date and install a few
packages that we need.

    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python-pip python-numpy python-scipy \
        libopenblas-dev libcupti-dev

Second, we install CUDA 8.0. It is publicly available on [Nvidia's website][2].

    wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo apt-get update
    sudo apt-get install cuda

Third, we install cuDNN 6.0, another library provided by Nvidia. Unfortunately,
this one is only accessible for registered members of [Nvidia's developers
program][3]. However, this program is free, and creating an account takes only
a couple of minutes. Once you are a member, you can find cuDNN [here][4].
Download it, and copy it to the EC2 instance. Then, execute the following
commands.

    tar xvf cudnn-8.0-linux-x64-v6.0.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/

We also need to modify an environment variable so that other software can pick
up where cuDNN is installed. In `~/.bashrc`, add the following line at the end
of the file.

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

Then, reload the file using `source ~/.bashrc`. Lastly, we install TensorFlow.

    sudo pip install tensorflow-gpu==1.4.0


## Running the collaborative RNN

We clone the repository and create a `data/` folder which will contain the
datasets.

    git clone https://github.com/lca4/collaborative-rnn.git
    cd collaborative-rnn/
    mkdir data


### Brightkite dataset

Downloading and preprocessing the data.

    cd data/
    wget https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz
    gunzip loc-brightkite_totalCheckins.txt.gz 
    cd ~/collaborative-rnn
    python utils/preprocess.py --output-dir data/ brightkite data/loc-brightkite_totalCheckins.txt

Running the collaborative RNN.

    python -u crnn.py data/brightkite-{train,valid}.txt --verbose \
        --hidden-size=32 --learning-rate=0.0075 --rho=0.997 \
        --chunk-size=64 --batch-size=20 --num-epochs=25

For this dataset, it should take less than a minute per epoch.

### Last.fm dataset

Downloading and preprocessing the data.

    cd data/
    wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
    tar -xvf lastfm-dataset-1K.tar.gz
    cd ~/collaborative-rnn

    # The next command takes ~6 minutes to complete.
    python utils/preprocess.py --output-dir data/ lastfm data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv

Running the collaborative RNN.

    python -u crnn.py data/lastfm-{train,valid}.txt --verbose \
        --hidden-size=128 --learning-rate=0.01 --rho=0.997 \
        --max-train-chunks=80 --max-valid-chunks=8 \
        --chunk-size=64 --batch-size=20 --num-epochs=10

For this dataset, it should take about 15 minutes per epoch.

[1]: https://aws.amazon.com/ec2/instance-types/p2/
[2]: https://developer.nvidia.com/cuda-downloads
[3]: https://developer.nvidia.com/
[4]: https://developer.nvidia.com/rdp/cudnn-download
