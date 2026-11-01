import json
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from modules.module8_answer_generation.validator import AnswerValidator
from modules.module8_answer_generation.evaluator import GenerationEvaluator

def run_tests():
    generator = GroundedAnswerGenerator(model_name="gemma3:4b")
    
    test_cases = [
        {
            "id": "Test 1 - Direct Factual Retrieval",
            "query": "What is the capital and largest city of France?",
            "evidence": [
                {
                    "title": "Paris",
                    "text": "Paris is the capital and most populous city of France, with an estimated population of over two million residents. Located along the Seine River in northern France, Paris is known for its culture and history."
                }
            ],
            "ground_truth": "Paris is the capital and most populous city of France."
        },
        {
            "id": "Test 2 - Multi-hop Chronological Comparison",
            "query": "Which magazine was published earlier, Arthur's Magazine or First for Women?",
            "evidence": [
                {
                    "title": "Arthur's Magazine",
                    "text": "Arthur's Magazine (1844-1846) was an American literary magazine published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar Allan Poe and others."
                },
                {
                    "title": "First for Women",
                    "text": "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989 and is based in Englewood Cliffs, New Jersey."
                }
            ],
            "ground_truth": "Arthur's Magazine was published earlier."
        },
        {
            "id": "Test 3 - Insufficient Evidence (Grounded Abstention)",
            "query": "Who won the 1994 FIFA World Cup?",
            "evidence": [
                {
                    "title": "History of Football",
                    "text": "Football is the world's most popular ball game in numbers of participants and spectators. Simple in its principal rules and essential equipment, the sport can be played almost anywhere."
                }
            ],
            "ground_truth": "Brazil"
        }
    ]

    print("=================================================================")
    print("MODULE 8 LIVE EVALUATION TEST REPORT (Model: gemma3:4b via Ollama)")
    print("=================================================================\n")

    for tc in test_cases:
        print(f"[{tc['id']}]")
        print(f"Query: {tc['query']}")
        res = generator.generate(
            query=tc["query"],
            evidence=tc["evidence"],
            ground_truth=tc.get("ground_truth")
        )
        print(f"Generated Answer: {res.final_answer}")
        print(f"Validation Status: {res.validation.status}")
        print(f"Is Grounded: {res.is_grounded} (Score: {res.grounding_score:.2f})")
        if res.evaluation:
            print(f"Exact Match: {res.evaluation.exact_match:.2f}")
            print(f"Token F1: {res.evaluation.f1_score:.2f}")
            print(f"Grounding Ratio: {res.evaluation.grounding_ratio:.2f}")
        print("-" * 65)

if __name__ == "__main__":
    run_tests()
