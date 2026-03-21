import json
import sys
from src.rag_pipeline import get_rag_answer

THRESHOLD = 0.5

def evaluate(answer, expected_keywords):
    score = 0
    for word in expected_keywords:
        if word.lower() in answer.lower():
            score += 1
    return score / len(expected_keywords)


print("🚀 Running Tests...\n")

with open("tests/test_cases.json") as f:
    test_cases = json.load(f)

for i, test in enumerate(test_cases):

    print(f"Test {i+1}: {test['question']}")

    answer = get_rag_answer(test["question"])
    print("Answer:", answer)

    score = evaluate(answer, test["expected_keywords"])
    print("Score:", score)

    if score < THRESHOLD:
        print("❌ FAILED")
        sys.exit(1)

    print("✅ PASSED\n")

print("🎉 All tests passed!")