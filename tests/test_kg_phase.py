'''Open Targets clinical stages, and which disease drugs the knowledge graph keeps.

Open Targets reports a marketed drug's `maxClinicalStage` as `APPROVAL`. If that maps to no
phase, the minimum-phase filter in `GetDiseaseAssociatedDrugs` drops every approved drug, and
an approved-only build (phase 4) gets an empty drug side. The cases below mix stages the API
returned for breast cancer with made-up edge cases; Open Targets itself is replaced with a stub.
'''

from __future__ import annotations

import backend.kgg.api_utils as kgg

STAGE_TO_PHASE = {
    "APPROVAL": 4,
    "APPROVED": 4,
    " approval\t": 4,
    " Approved ": 4,
    "PHASE_3": 3,
    # Combined stages count as the lower phase, as ChEMBL's max_phase does.
    "PHASE_2_3": 2,
    "PHASE_2": 2,
    "PHASE_1_2": 1,
    "PHASE_1": 1,
    "EARLY_PHASE_1": 1,
    "PRECLINICAL": None,
    "UNKNOWN": None,
    "PREAPPROVAL": None,
    "PREAPPROVED": None,
    "NOT_APPROVED": None,
    "": None,
    None: None,
}


def test_open_targets_stages_map_to_phases() -> None:
    for stage, phase in STAGE_TO_PHASE.items():
        result = kgg._clinical_stage_to_phase(stage)
        assert result == phase, f"{stage!r} -> {result}, expected {phase}"


def _candidates(*stages: str) -> dict:
    return {
        "disease": {
            "id": "MONDO_0007254",
            "name": "breast cancer",
            "drugAndClinicalCandidates": {
                "count": len(stages),
                "rows": [
                    {
                        "id": f"row{i}",
                        "maxClinicalStage": stage,
                        "drug": {"id": f"CHEMBL{i}", "name": stage, "drugType": "Small molecule"},
                        "clinicalReports": [],
                    }
                    for i, stage in enumerate(stages)
                ],
            },
        }
    }


def test_approved_drugs_survive_the_phase_filter() -> None:
    payload = _candidates("APPROVAL", "PHASE_3", "PHASE_1_2")
    original_query, original_count = kgg._opentargets_query, kgg.getDrugCount
    kgg._opentargets_query = lambda query, variables: payload
    kgg.getDrugCount = lambda disease_id: 3
    try:
        approved_only = kgg.GetDiseaseAssociatedDrugs("MONDO_0007254", 4)
        default = kgg.GetDiseaseAssociatedDrugs("MONDO_0007254", 3)
    finally:
        kgg._opentargets_query, kgg.getDrugCount = original_query, original_count

    assert list(approved_only["drugId"]) == ["CHEMBL0"]
    assert list(approved_only["status"]) == ["APPROVAL"]
    assert list(default["drugId"]) == ["CHEMBL0", "CHEMBL1"]


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
