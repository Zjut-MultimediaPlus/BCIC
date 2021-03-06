# Gcn-ConCap: Boosting Convolutional Image Captioning with Semantic Content and Visual Realationship

PyTorch implementation of -- [Boosting Convolutional Image Captioning with Semantic Content and Visual Realationship]                 

Clone the repository with the --recursive flag to recursively clone third party submodules. 
For example,

```
git clone --recursive https://github.com/aditya12agd5/convcap.git
```

For setup first install [PyTorch-0.2.0_3](http://pytorch.org/). For this code we used cuda-8.0, 
python-2.7 and pip

```
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
```

torchvision-0.1.9 was installed from [source](https://github.com/pytorch/vision/releases)

Install other python packages using 

```
pip install -r requirements.txt
```

A wordlist is provided in ./data/wordlist.p

Fetch the train/val/test splits (same as [NeuralTalk](http://cs.stanford.edu/people/karpathy/deepimagesent/)) 
for MSCOCO with

```
bash scripts/fetch_splits.sh
```

Download train2014, val2014 images and their annotations from the [MSCOCO](http://cocodataset.org/#download) 
webpage and put them in ./data/coco
 
To train the model on MSCOCO from scratch,

```
python main.py model_dir
```

model_dir is the directory to save model & results. Run python main.py -h for details about other
command line arguments. Two models will be saved, model.pth at the end of every epoch and 
bestmodel.pth, the model that obtains best score (on CIDEr metric by default) over all epochs. 
 
To train the model without attention use the --no-attention flag,

```
python main.py --no-attention model_dir
```

To test on MSCOCO with the released model,

```
python main.py -t 0 model_dir
```

model_dir should contain the released model bestmodel.pth. Run, scripts/fetch_trained_model.sh, 
it will store the trained bestmodel.pth in ./data/

To caption your own images,

```
python captionme.py model_dir image_dir
```

model_dir should contain the released model bestmodel.pth. Captions for *png, *jpg images in
image_dir will be saved in image_dir/captions.txt. Run, python captionme.py -h for additional
options 


