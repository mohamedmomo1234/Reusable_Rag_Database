import os

from graph import ask_rag_bot


TEST_CASES = [
    item.strip()
    for item in os.getenv("EVAL_QUESTIONS", "").split("|")
    if item.strip()
]


def run_evaluation():
    if not TEST_CASES:
        print(
            "No EVAL_QUESTIONS configured. "
            "Set EVAL_QUESTIONS as questions separated by |."
        )
        return []

    results = []

    for question in TEST_CASES:
        try:
            result = ask_rag_bot(question)
            has_context = bool(result.get("sources"))

            results.append(
                {
                    "question": question,
                    "has_sources": has_context,
                    "sources": result.get("sources", []),
                    "answer": result.get("answer", ""),
                }
            )

        except Exception as exc:
            results.append(
                {
                    "question": question,
                    "has_sources": False,
                    "sources": [],
                    "answer": f"ERROR: {exc}",
                }
            )

    coverage = (
        sum(item["has_sources"] for item in results)
        / len(results)
        if results
        else 0
    )

    print(f"Retrieval coverage: {coverage:.2%}")

    for item in results:
        status = "PASS" if item["has_sources"] else "FAIL"

        print(
            f"{status} | {item['question']} | "
            f"{item['sources']}"
        )

    return results


if __name__ == "__main__":
    run_evaluation()
