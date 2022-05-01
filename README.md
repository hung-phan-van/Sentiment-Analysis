# Sentiment-analysis


## Set up environment
Installing miniconda for environment variable control:
```
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Creating conda envirenment
```
conda create -n scene python=3.8
conda activate scene 
pip install --upgrade pip
```
**Please make sure that the python version >= 3.6**
## Installing basic packages and AI packages
Clone the repository and enter to root folder of it:
```
git clone <gitlab_link>
cd <project_directory>
```
Installing basic packages:
```
pip install -r requirements.txt
```
Installing AI packages:
```
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Please note that these AI packages are **only compatible with CUDA 11.1**

