# Documentation for installing dgl python package 
I am not sure the reason but I have tried different methods of downloading and installing dgl on Franklin server provided by stat.lab.wisc.edu but none of these could work. Thus, here is a documentation showing what I have tried (beyond what was shown here) and if it worked. 


# Failed trial 1 --> Doesn't work at all. 
This method follows someone's post I cannot find anymore. 
However, this method doesn't work at all. 
```
conda create -n dgl python=3.11
source activate myenv
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

## Install dgl which matches pytorch 2.2 and cuda 12.1 
conda install -c dglteam/label/cu121 dgl

## Add environment to jupyter kernel
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=myenv

# install remaining things that dgl needs
pip install torchdata
pip install pandas
pip install pyyaml
pip install pydantic

```

# Failed trial 2: 
Follows information here: https://discuss.dgl.ai/t/importerror-cannot-load-graphbolt-c-library/4291/18

```
conda create -n dgl2.1 python=3.11 
conda dactivate dgl2.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
python -c "import torch; print(torch.__version__)" # printed out 2.2.0+cu121 

pip install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
python -c "import dgl; print(dgl.__version__)" 
# This gave out this error which has appeared many times: 
# ModuleNotFoundError: No module named 'torchdata.datapipes'
# Thus, uninstall torch first: 

# Try to install pytorch 2.1.0 
# CPU only 
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
export PATH="/u/b/i/bingl/miniconda3/envs/dgl2.1/bin:$PATH"
python -c "import torch; print(torch.__version__)"  
# error message suggests that I need to downgrade numpy 

pip install numpy==1.26.4 --force-reinstall

python -c "import torch; print(torch.__version__)"  #finally 2.1.0 

pip install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
python -c "import dgl; print(dgl.__version__)"
# another error "OSError: libcudart.so.12: cannot open shared object file: No such file or directory", so: 

# conda install -c conda-forge cudatoolkit=12.1 --> Doesn't work 
# conda install -c nvidia cudatoolkit-dev=12.1 --> Doesn't work 

pip install nvidia-pyindex
pip install nvidia-cuda-runtime-cu12

python -c "import torch; print(torch.version.cuda)"
```

# Option 1: 
Install Dgl for cpu only 
```
conda create -n dgl-cpu python=3.11 -y 
conda activate dgl-cpu




```


# Option 2: 
This follows the installation instruction of dgl: https://www.dgl.ai/dgl_docs/install/index.html

```
git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl 
bash script/create_dev_conda_env.sh -c



```