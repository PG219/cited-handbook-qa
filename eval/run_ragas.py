from ragas import evaluate
from ragas.metrics.collections import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

from datasets import Dataset
import json

with open("eval/ragas_input.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)

print(results)
