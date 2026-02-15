"""Summarization tests — run with: python -m tests.test_summarize"""

import time
from model_loader import count_tokens, tokenizer, model, is_seq2seq, is_t5_based
from summarizer import summarize_text, calculate_generation_params
from config import MODEL_NAME

TEST_TEXT = """
Python is a high-level, interpreted, general-purpose programming language.
Its design philosophy emphasizes code readability with the use of significant indentation.
Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms,
including structured (particularly procedural), object-oriented and functional programming.

Python was created in the late 1980s by Guido van Rossum as a successor to the ABC programming language.
Python 2.0, released in 2000, introduced features like list comprehensions and a garbage collection system.
Python 3.0, released in 2008, was a major revision that is not completely backward-compatible with earlier versions.

Python is consistently ranked as one of the most popular programming languages. It is widely used in
web development, data science, machine learning, automation, and scientific computing. Major organizations
like Google, NASA, and CERN use Python extensively in their operations.

The language has a comprehensive standard library and supports various third-party packages through the
Python Package Index (PyPI). Popular frameworks include Django and Flask for web development, NumPy and
Pandas for data analysis, and TensorFlow and PyTorch for machine learning applications.
"""


def test_summarization():
    print(f"\nModel: {MODEL_NAME}, seq2seq: {is_seq2seq}, T5: {is_t5_based}")
    print(f"Input: {count_tokens(TEST_TEXT)} tokens\n")

    for target in [100, 150, 200]:
        start = time.time()
        summary = summarize_text(TEST_TEXT, target)
        duration = time.time() - start

        tokens = count_tokens(summary)
        words = len(summary.split())
        model_max = getattr(tokenizer, "model_max_length", 1024)
        _, min_new, _ = calculate_generation_params(target, model_max)

        print(f"target={target}: {tokens} tokens, {words} words, {duration:.2f}s")
        print(f"  {summary[:200]}{'...' if len(summary) > 200 else ''}")
        print(f"  Multi-sentence: {words > 5}, Target met: {min_new <= tokens <= target * 1.2}\n")


if __name__ == "__main__":
    test_summarization()
