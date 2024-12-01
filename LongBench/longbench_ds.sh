model="llama3.1-8b-instruct-128k"


for task in "musique" "narrativeqa" "qasper"
do
    for group_factor in 2 4 8
    do
        for q_bits in 2 4 8
        do
            for budget in 64 128 256 512 # 1024 2048 4096
            do
                python -u pred.py \
                    --model $model --task $task \
                    --ds --heavy_const $budget --group_factor 2 --q_bits 2 --channel q
            done
        done
    done
done

# python -u eval.py --model $model
