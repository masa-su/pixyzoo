for seed in `seq 100 200`
do 
    for lr in 0.001 0.0001
    do
        python vrnn_model_f.py --lr $lr --seed $seed
        python vrnn.py --lr $lr --seed $seed
    done
done