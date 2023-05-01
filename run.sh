for model in resnet18
do
    for dataset in FashionMNIST
    do
        for bs in 128
        do
            for lr in 0.01
            do
                for epochs in 300
                do
                    for seed in 0 1 2
                    do
                        for cuda in 0
                        do
                            nohup python train.py       \
                            --model_mode    $model      \
                            --dataset       $dataset    \
                            --bs            $bs         \
                            --lr            $lr         \
                            --epochs        $epochs     \
                            --seed          $seed       \
                            --cuda          $cuda       \
                            --use_lr_sche > $model-$dataset-bs$bs-lr$lr-seed$seed.out 2>&1 &
                        done
                    done
                done
            done
        done
    done
done