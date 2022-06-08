## An Implementation of GIRAFFE

GIRAFFE (Representing Scenes as Compositional Generative Neural Feature Fields) project website: [here](https://m-niemeyer.github.io/project-pages/giraffe/index.html). Please read their original github code for more details. 

### Installation

It is suggested to have Python>=3.8 
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

### Datasets

To train a model from scratch or to use their ground truth activations for evaluation, you have to download the respective dataset.

For this, please run
```
bash scripts/download_dataset.sh
```
and following the instructions. This script should download and unpack the data automatically into the `data/` folder.


### Controllable Image Synthesis

To render images of a trained model, run
```
python render.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.
The easiest way is to use a pre-trained model.
You can do this by using one of the config files which are indicated with `*_pretrained.yaml`. 

For example, for theirr model trained on Cars at 256x256 pixels, run
```
python render.py configs/256res/cars_256_pretrained.yaml
```
or for celebA-HQ at 256x256 pixels, run
```
python render.py configs/256res/celebahq_256_pretrained.yaml
```
Their script will automatically download the model checkpoints and render images.
You can find the outputs in the `out/*_pretrained` folders.

### Training
Finally, to train a new network from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs
```
where you replace `OUTPUT_DIR` with the respective output directory. For available training options, please take a look at `configs/default.yaml`.

## Using Your Own Dataset

If you want to train a model on a new dataset, you first need to generate ground truth activations for the intermediate or final FID calculations.
For this, you can use the script in `scripts/calc_fid/precalc_fid.py`.
For example, if you want to generate an FID file for the comprehensive cars dataset at 64x64 pixels, you need to run
```
python scripts/precalc_fid.py  "data/comprehensive_cars/images/*.jpg" --regex True --gpu 0 --out-file "data/comprehensive_cars/fid_files/comprehensiveCars_64.npz" --img-size 64
```
or for LSUN churches, you need to run
```
python scripts/precalc_fid.py path/to/LSUN --class-name scene_categories/church_outdoor_train_lmdb --lsun True --gpu 0 --out-file data/church/fid_files/church_64.npz --img-size 64
```

Note: We apply the same transformations to the ground truth images for this FID calculation as we do during training. If you want to use your own dataset, you need to adjust the image transformations in the script accordingly. Further, you might need to adjust the object-level and camera transformations to your dataset. 
