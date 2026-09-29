"""
Optional RAGAS evaluation.

This script expects a CSV named ragas_dataset.csv with:
question,answer,ground_truth

Example:
question,answer,ground_truth
"What is diabetes?","...","..."

RAGAS APIs change between releases. If your installed RAGAS version has
different metric/import names, check the version-specific RAGAS docs before
running this file.
"""

import pandas as pd


def build_dataset(path="ragas_dataset.csv"):
    return pd.read_csv(path)


def run_ragas(path="ragas_dataset.csv"):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    from graph import ask_medical_bot

    df = build_dataset(path)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for _, row in df.iterrows():
        question = str(row["question"])
        result = ask_medical_bot(question)

        questions.append(question)
        answers.append(result["answer"])
        ground_truths.append(str(row["ground_truth"]))

        # RAGAS needs retrieved contexts, so retrieve separately.
        from retriever import retrieve_documents

        docs = retrieve_documents(question)
        contexts.append([doc.page_content for doc in docs])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print(result)
    return result


if __name__ == "__main__":
    run_ragas()
