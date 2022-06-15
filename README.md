## An Implementation of GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields

GIRAFFE (Representing Scenes as Compositional Generative Neural Feature Fields) project website: [here](https://m-niemeyer.github.io/project-pages/giraffe/index.html). Please read the original github code for more details. 

### Installation

It is suggested to have Python >= 3.8 
```angular2html
sudo apt install python3.8
sudo apt install python3.8-distutils
```

```angular2html
git clone https://github.com/YHDING23/Demo_giraffe.git
cd Demo_giraffe
mkdir venv_giraffe
virtualenv -p python3.8 venv_giraffe
source venv_giraffe/bin/activate

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Quick Start
You can now test our code on the provided pre-trained models.
For example, simply run
```
python render.py configs/256res/cars_256_pretrained.yaml
```
This script should create a model output folder `out/cars256_pretrained`.
The animations are then saved to the respective subfolders in `out/cars256_pretrained/rendering`. It contains at least 3 subfolders for rotation, interpolation on appearence, and interpolation on shape. 

The folder `configs` provides multiple options for using their pretrained models. Please note that the config files  `*_pretrained.yaml` are only for evaluation or rendering, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pre-trained model. 


### Controllable Image Synthesis

To render images of a trained model, run
```
python render.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.
The easiest way is to use a pre-trained model.
You can do this by using one of the config files which are indicated with `*_pretrained.yaml`. 

For example, for our model trained on celebA-HQ at 256x256 pixels, run
```
python render.py configs/256res/celebahq_256_pretrained.yaml
```
Our script will automatically download the model checkpoints and render images.
You can find the outputs in the `out/*_pretrained` folders.

### Training

To train a model from scratch or to use your ground truth activations for evaluation, you have to download the respective dataset.

For this, please run
```
bash scripts/download_dataset.sh
```
and following the instructions. This script should download and unpack the data automatically into the `data/` folder. 
```angular2html
0 - Cats Dataset
1 - CelebA Dataset
2 - Cars Dataset
3 - Chairs Dataset
4 - Church Dataset
5 - CelebA-HQ Dataset
6 - FFHQ Dataset
7 - Clevr2 Dataset
8 - Clevr2345 Dataset
```
For example, download `CelebA` dataset to enter `1`.

Then, to train a new network from scratch, run
```
python train.py configs/64res/celeba_64.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd out/celeba64
tensorboard --logdir logs
```

Make the remote server accept your local external IP by running:
```angular2html
ssh -L 16006:127.0.1:6006 your_user_ID@remote_server_IP
```
It forwards everything on the port 6006 of the remote server (in 127.0.0.1:6006) to your local machine on the port 16006. On your local machine, go to http://127.0.0.1:16006 and enjoy the remote TensorBoard. 
