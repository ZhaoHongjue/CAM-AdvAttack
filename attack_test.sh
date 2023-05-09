for method in CW
do
    for model_mode in densenet121
    do
        for dataset in Imagenette
        do
            for seed in 0 1 2
            do
                python attack_test.py               \
                --method            $method         \
                --model_mode       	$model_mode	    \
                --dataset           $dataset        \
                --cuda              3               \
                --seed $seed
                echo "Seed$seed Done!"
            done
        done
    done
done