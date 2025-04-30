export CUDA_VISIBLE_DEVICES=0

python test.py \
resnet "MNIST_result/model_params/ResNet_relu_L2-0.0_lr-0.01_augment=True_schduler=MultiStepLR_[2000, 4000]_0.2.npz" \
--layer-size 600 100 \
--batch-size 256 \
--activ-func relu \