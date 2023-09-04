### Install wsl with powershell 
- wsl --install 
- Note: you may have to enable virtualization on your PC 

### WSL - Once inside of linux: 
- sudo apt update
- sudo apt-get upgrade

### WSL - Install miniconda: 
- https://javedhassans.medium.com/install-miniconda-on-linux-from-the-command-line-in-5-steps-403912b3f378:
- mkdir -p ~/miniconda3
- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
- bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
- ~/miniconda3/bin/conda init bash
- restart the shell 
- conda list

### Windows - Install Mangio RVC: 
- Download [7zip](https://www.7-zip.org/)
- Follow these instructions to download one of the .7z files. You can use either INFER or INFERTRAIN depending on your use case [Mangio RVC 7zip install guide](https://docs.google.com/document/d/1KKKE7hoyGXMw-Lg0JWx16R8xz3OfxADjwEYJTqzDO1k/edit)
- I downloaded the Mangio-RVC-v23.7.0_INFER_TRAIN.7z to my **windows** machine and then unzipped the zip with 7z to my downloads folder. Keep note where this is we will copy it over to the Unbuntu WSL install in a second 

### WSL - Clone the pipeline repo:
- git clone https://github.com/DrewScatterday/tortoise_MangioRVC.git

### Windows - Copy Mangio RVC over 
- Use file explorer and scroll all the way down on the left quick access bar to find "Linux" 
- Open the "Ubuntu" folder and then you can now easily copy and paste files from Windows to WSL 
- Copy the "Mangio-RVC-v23.7.0" folder you unzipped on windows and put inside of the tortoise_MangioRVC directory on WSL 

### WSL - Clone the tortoise fast repo:
- cd into tortoise_MangioRVC
- git clone https://github.com/DrewScatterday/tortoise-tts-fast.git
- conda create --name tortoiseRVC python=3.9 numba inflect
- conda activate tortoiseRVC
- conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- conda install transformers=4.29.2
- conda install -c conda-forge cudatoolkit-dev
- sudo apt-get install gcc
- sudo apt-get install g++
- pip install -r requirements.txt
- pip3 install git+https://github.com/152334H/BigVGAN.git
- pip install deepspeed==0.10.2 

### WSL - Open tortoise_MangioRVC in VS code: 
- You can use VS code on Windows and link up to WSL to view files and edit the code 
- You'll need to edit the `pipeline.py` code to have valid paths 
- You'll also need to copy any .pth files or index files in their respective directories 
- You may need to edit the path of MANGIO_DIR `rvc_infer.py` 
- Run python `pipeline.py` 
