model="llama3.1-8b-instruct-128k"

for task in "musique" "narrativeqa" "qasper"
do
    for budget in 64 128 256 512 1024 2048 4096
    do
        python -u pred.py \
            --model $model --task $task \
            --topk --token_budget $budget
    done
done

# python -u eval.py --model $model
