# PreMix

## Requirements

Install pytorch:

```python
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other required packages:

```python
pip install -r requirements.txt
```

## Training

We use [NBU_PansharpRSData](https://github.com/starboot/NBU_PansharpRSData) for training and testing.

One training example:
```python
python train.py --rgb_c 2,1,0 --ms_chans 4 --sensor GF --data_dir "~/data/3 Gaofen-1" --embed_dim 32 --kernel_size 3 --pf_kernel 3 --num_layers 1 --EWFM --activation tanh+relu
```


## Testing

The training process will obtain a checkpoint with best performence at validation set. One testing example:

```python
python test.py --rgb_c 2,1,0 --ms_chans 4 --sensor GF --data_dir "~/data/3 Gaofen-1" --embed_dim 32 --kernel_size 3 --pf_kernel 3 --num_layers 1 --EWFM --activation tanh+relu --ckpt log_m=PreMixHuge_s=GF_l=1_d=32_k=3_pfk=3_EWFM=True_a=tanh+relu/ep=279_PSNR=44.7997.ckpt
```