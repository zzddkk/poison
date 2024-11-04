# poison
## Set Up
1. create a virtual environment by conda
```
conda create -n poison python=3.10.12 -y
```
2. install the packages
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (For cuda 12.1)
pip install PyWavelets==1.7.0
pip install scipy==1.14.1
pip install opencv-python==4.10.0.84
pip install av==13.1.0
pip install pillow==10.2.0
pip install WeTextProcessing==1.0.4.1
pip install tqdm==4.66.6
pip install numpy==1.26.3
conda install ffmpeg
```
## word_boundary
Get the word_boundary [wb](./word_boundary)
## The poison dict need
we need the poison dict structure as following
```
{
    "id1":
        {
            "word1":
                {
                    "pert": int,
                    "X"   : list,
                    "Y"   : list,
                    "F_lower" : int,
                    "F_upper" " int
                    "poison_style" str (the value is select from "FFT"/"DWT"/"RANDOM"/"DCT")
                }
        },
    "id2",
        {

        },
    ...
}
```
