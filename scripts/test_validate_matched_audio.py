from copy import deepcopy

from validate_matched_audio import ARMS, validate_pair


def fixture():
    records = {}
    for arm, system in ARMS.items():
        records[arm] = dict(system=system, example_index=0,
            reference_item_ids=[str(i) for i in range(15)], target_item_id="heldout",
            retrieved_item_id="predicted", neighbor_item_ids=["neighbor"], style_item_ids=["style"],
            music_attributes="attributes", lyric_draft="lyrics",
            cue_ids=list(range(8)) if arm == "full" else [],
            cue_terms=[f"cue{i}" for i in range(8)] if arm == "full" else [])
    return records


def test_matched_inputs_and_intervention():
    records = fixture()
    validate_pair(records, 0)
    for field, value in [("reference_item_ids", ["other"] * 15),
                         ("neighbor_item_ids", ["different"]), ("cue_terms", ["leak"])]:
        broken = deepcopy(records)
        broken["drop"][field] = value
        try:
            validate_pair(broken, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Mismatch accepted: {field}")


if __name__ == "__main__":
    test_matched_inputs_and_intervention()
    print("PASS matched histories, unchanged neighbors and no cue leakage")
