from __future__ import annotations


from semblo.tasks.longbench_v2 import LongBenchV2Task


def test_longbench_v2_streaming_flag_and_official_prompt(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_name, *, split, streaming):
        calls.append((dataset_name, split, streaming))
        return [
            {
                "_id": "x",
                "domain": "single-doc",
                "sub_domain": "qa",
                "difficulty": "easy",
                "length": "short",
                "question": "Which choice?",
                "choice_A": "A1",
                "choice_B": "B1",
                "choice_C": "C1",
                "choice_D": "D1",
                "answer": "A",
                "context": "context",
            }
        ]

    monkeypatch.setattr(
        "semblo.tasks.longbench_v2.load_dataset",
        fake_load_dataset,
    )

    examples = list(
        LongBenchV2Task(
            {
                "dataset_name": "demo/lb2",
                "split": "train",
                "streaming": True,
            }
        ).iter_examples()
    )

    assert calls == [("demo/lb2", "train", True)]
    assert examples[0].id == "longbench_v2/x"
    assert examples[0].metadata["bin"] == "short"
    assert "The correct answer is (insert answer here)" in (
        examples[0].messages[0]["content"]
    )


def test_longbench_v2_filters_domain_and_length(monkeypatch):
    rows = [
        {
            "_id": "keep",
            "domain": "multi-doc",
            "sub_domain": "qa",
            "difficulty": "hard",
            "length": 64000,
            "question": "Q",
            "choice_A": "A",
            "choice_B": "B",
            "choice_C": "C",
            "choice_D": "D",
            "answer": "A",
            "context": "ctx",
        },
        {
            "_id": "drop",
            "domain": "single-doc",
            "sub_domain": "qa",
            "difficulty": "easy",
            "length": 16000,
            "question": "Q",
            "choice_A": "A",
            "choice_B": "B",
            "choice_C": "C",
            "choice_D": "D",
            "answer": "A",
            "context": "ctx",
        },
    ]
    monkeypatch.setattr(
        "semblo.tasks.longbench_v2.load_dataset",
        lambda *args, **kwargs: rows,
    )

    task = LongBenchV2Task(
        {
            "domains": ["multi-doc"],
            "difficulties": ["hard"],
            "max_words": 100000,
        }
    )

    assert [example.id for example in task.iter_examples()] == [
        "longbench_v2/keep"
    ]
