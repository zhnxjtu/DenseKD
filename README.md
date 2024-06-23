# DenseKD
Submitted to TNNLS (Under Review)


#### To achieve the lightweight ResNet-56
python train.py --model resnet20 --teacher resnet56 --teacher-weight ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --kd-loss-weight 1.8 --cuda-idx 0 --instance-tmp 0.75 --patch-size 8 --suffix densekd1
