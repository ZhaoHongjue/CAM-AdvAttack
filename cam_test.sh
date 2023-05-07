for seed in 0 1 2
do
    python cam_test.py                  \
    --method            all             \
    --model_mode       	densenet121        \
    --dataset           Imagenette    \
    --seed $seed
    echo "Seed$seed Done!"
done
