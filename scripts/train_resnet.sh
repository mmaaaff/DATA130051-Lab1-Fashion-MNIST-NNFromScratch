export CUDA_VISIBLE_DEVICES=0

python train.py resnet adam 256 30 \
--scheduler MultiStepLR \
--milestones 2000 4000 \
--scheduler-gamma 0.2 \
-lr 0.01 \
--lambda-L2 0.0 \
--augment True \
--augment-prob 0.5 \
--val-interval 30 \
--model-path MNIST_result/model_params --result-path MNIST_result/results