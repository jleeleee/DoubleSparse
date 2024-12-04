model="llama3.1-8b-instruct-128k"


for task in "musique" "narrativeqa" "qasper"
do
    for budget in 64 128 256 512 # 1024 2048 4096
    do
        for q_bits in 4 8
        do
            for group_factor in 4 8
            do
                python -u pred.py \
                    --model $model --task $task \
                    --ds --heavy_const $budget --group_factor $group_factor --q_bits $q_bits --channel q
            done
        done
    done
done

# python -u eval.py --model $model
