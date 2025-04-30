export CUDA_VISIBLE_DEVICES=0

python train.py mlp adam 256 30 \
--scheduler None \
--milestones 500 1000 2000 \
--scheduler-gamma 0.3 \
-lr 0.05 \
--lambda-L2 0.5 \
--augment False \
--augment-prob 0.5 \
--val-interval 30 \
--layer-size 600 100 \
--mlp-dropout 0.7 \
--model-path MNIST_result/model_params --result-path MNIST_result/results