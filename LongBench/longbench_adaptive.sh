model="llama3.1-8b-instruct-128k"

for task in "qasper" # "narrativeqa" # "musique"
do
    for budget in 1024 2048 4096 64 128 256 512
    do
        python -u pred.py \
            --model $model --task $task \
            --adaptive --token_budget $budget
    done
done

# python -u eval.py --model $model
