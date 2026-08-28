"""
Orchestrates the full pipeline for one or more candidates:

  PDFs -> profile_builder -> agents (independent) -> debate -> decision -> report

Run:
    python main.py
"""

import json
import os
from pathlib import Path

from pypdf import PdfReader

from profile_builder import build_profile
from agents import get_independent_opinions
from debate import run_full_debate
from decision import make_final_decision

DATA_DIR = Path("data")          # put your PDFs here
OUTPUT_DIR = Path("output")      # full run logs + reports land here


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def run_pipeline_for_candidate(candidate_id: str, resume_path: Path,
                                transcript_path: Path, job_description: str) -> dict:
    print(f"[{candidate_id}] extracting text from PDFs...")
    resume_text = extract_pdf_text(resume_path)
    transcript_text = extract_pdf_text(transcript_path)

    print(f"[{candidate_id}] building candidate profile...")
    profile = build_profile(resume_text, transcript_text)

    print(f"[{candidate_id}] running 4 independent agents (parallel, isolated)...")
    initial_opinions = get_independent_opinions(profile, job_description)

    print(f"[{candidate_id}] running debate...")
    debate_result = run_full_debate(initial_opinions, rounds=2)

    print(f"[{candidate_id}] making final decision...")
    decision = make_final_decision(initial_opinions, debate_result)

    run_record = {
        "candidate_id": candidate_id,
        "profile": profile,
        "initial_opinions": initial_opinions,
        "debate_final_opinions": debate_result["final_opinions"],
        "change_log": debate_result["change_log"],
        "final_decision": decision
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{candidate_id}_full_run.json"
    with open(out_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"[{candidate_id}] full run log saved to {out_path}")

    return run_record


def print_report(run_record: dict):
    d = run_record["final_decision"]
    print("\n" + "=" * 60)
    print(f"FINAL REPORT — {run_record['candidate_id']}")
    print("=" * 60)
    print(f"Recommendation: {d['recommendation']}  (confidence {d['confidence']})")
    print(f"\nRationale:\n{d['weighting_rationale']}")

    print("\nStrengths:")
    for s in d["strengths"]:
        print(f"  + {s['point']}  —  \"{s['quote_or_fact']}\"  [{s['raised_by']}]")

    print("\nConcerns:")
    for c in d["concerns"]:
        print(f"  - {c['point']}  —  \"{c['quote_or_fact']}\"  [{c['raised_by']}]")

    if d["unresolved_disagreements"]:
        print("\nUnresolved disagreements:")
        for u in d["unresolved_disagreements"]:
            print(f"  ! {u}")

    if run_record["change_log"]:
        print("\nOpinion changes during debate:")
        for c in run_record["change_log"]:
            print(f"  round {c['round']}: {c['agent']} {c['old_verdict']} -> {c['new_verdict']} "
                  f"(triggered by {c['triggered_by']})")
    print()


if __name__ == "__main__":
    job_description = extract_pdf_text(DATA_DIR / "02_Job_Description.pdf")

    candidates = [
        ("Candidate_A", DATA_DIR / "03_Resume_A.pdf", DATA_DIR / "05_Transcript_A.pdf"),
        ("Candidate_B", DATA_DIR / "04_Resume_B.pdf", DATA_DIR / "06_Transcript_B.pdf"),
    ]

    for candidate_id, resume_path, transcript_path in candidates:
        record = run_pipeline_for_candidate(candidate_id, resume_path, transcript_path, job_description)
        print_report(record)
