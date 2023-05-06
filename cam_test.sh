for seed in 0 1 2
do
    python cam_test.py                  \
    --method            all             \
    --model_mode        resnet18        \
    --dataset           CIFAR10    \
    --seed $seed
    echo "Seed$seed Done!"
done