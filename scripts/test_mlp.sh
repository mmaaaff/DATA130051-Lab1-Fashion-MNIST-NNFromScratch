export CUDA_VISIBLE_DEVICES=0

python test.py \
mlp "MNIST_result/model_params/MLP3_(600,100)_dropout0.7_relu_L2-0.5_lr-0.05_augment=False_optim=adam_schduler=None_[500, 1000, 2000]_0.3.npz" \
--layer-size 600 100 \
--batch-size 256 \
--activ-func relu \