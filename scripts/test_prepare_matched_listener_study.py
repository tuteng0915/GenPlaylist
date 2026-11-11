from prepare_listener_study import _select_indices
from prepare_matched_listener_study import assignments


def test_stable_balanced_blinding_and_separate_case_namespaces():
    indices = _select_indices(941, 25, 42)
    first = assignments(indices, 42, "drop")
    assert first == assignments(indices, 42, "drop")
    assert sum(x["full_is_a"] for x in first) == 13
    assert len(set(x["case_id"] for x in first)) == 25
    second = assignments(indices, 42, "sft")
    assert not set(x["case_id"] for x in first) & set(x["case_id"] for x in second)


if __name__ == "__main__":
    test_stable_balanced_blinding_and_separate_case_namespaces()
    print("PASS deterministic 25-case blinding, 13/12 sides and separate study IDs")
