model="llama3.1-8b-instruct-128k"

for task in "narrativeqa"
do
    python -u pred.py \
        --model $model --task $task \
        --topk

done

# python -u eval.py --model $model
