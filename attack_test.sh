for method in LBFGS
do
    for model_mode in densenet121
    do
        for dataset in Imagenette
        do
            for seed in 0
            do
                python attack_test.py               \
                --method            $method         \
                --model_mode       	$model_mode	    \
                --dataset           $dataset        \
                --seed $seed
                echo "Seed$seed Done!"
            done
        done
    done
done