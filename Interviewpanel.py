"""Run: pip install -r requirements.txt
PowerShell: $env:OPENAI_API_KEY="your key"
python interview_panel.py job.pdf resume.pdf transcript.pdf candidate_a
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
from pypdf import PdfReader

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
client = OpenAI()
PERSONAS = {
    "technical": "Assess technical capability, depth, and relevance to the role.",
    "hr_culture": "Assess communication, teamwork, professionalism, and honesty. Never infer protected traits.",
    "hiring_manager": "Assess likelihood of delivering in this exact role.",
    "skeptic": "Find contradictions, exaggeration, unsupported claims, and material risks.",
}


def pdf_text(path: str) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def call_json(instructions: str, prompt: str) -> dict[str, Any]:
    response = client.responses.create(model=MODEL, instructions=instructions + " Return valid JSON only, with no Markdown.", input=prompt)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.output_text.strip())
    return json.loads(text)


def evidence_errors(value: Any, source: str, key: str = "") -> list[str]:
    """Reject displayed evidence that is not a verbatim quote from a supplied document."""
    if isinstance(value, dict):
        return sum((evidence_errors(v, source, k) for k, v in value.items()), [])
    if isinstance(value, list):
        return sum((evidence_errors(item, source, key) for item in value), [])
    return [value] if key == "evidence" and isinstance(value, str) and value not in source else []


def profile_prompt(sources: str) -> str:
    return f'''Extract only facts from the following resume and interview transcript. Do not evaluate the candidate.
{sources}

Return this JSON: {{"skills_claimed":[{{"fact":"...","evidence":["exact quote"]}}],"experience":[{{"fact":"...","evidence":["exact quote"]}}],"interview_claims":[{{"fact":"...","evidence":["exact quote"]}}],"unknowns":["..."],"possible_contradictions":[{{"description":"...","evidence":["quote 1","quote 2"]}}]}}.
Every fact needs an exact quote. Put unsupported items in unknowns.'''


def independent_prompt(role: str, duty: str, job: str, profile: dict[str, Any]) -> str:
    return f'''You are the {role} panelist. {duty}
JOB DESCRIPTION:\n{job}\nSOURCE-DERIVED PROFILE:\n{json.dumps(profile, ensure_ascii=False)}
You are working independently and have not seen any other agent's opinion.
Return JSON: {{"agent":"{role}","initial_recommendation":"hire|no_hire|hold","confidence":"low|medium|high","strengths":[{{"claim":"...","evidence":["exact quote"]}}],"concerns":[{{"claim":"...","severity":"low|medium|high","evidence":["exact quote"]}}],"unknowns":["..."],"questions_for_debate":["..."]}}.
Do not score. Each strength or concern needs a quote from the profile; state unknown when there is insufficient evidence.'''


def debate_prompt(role: str, original: dict[str, Any], opinions: dict[str, Any], source: str) -> str:
    others = {name: opinion for name, opinion in opinions.items() if name != role}
    return f'''You are the {role} panelist in the debate.
YOUR ORIGINAL OPINION: {json.dumps(original, ensure_ascii=False)}
OTHER ORIGINAL OPINIONS: {json.dumps(others, ensure_ascii=False)}
SOURCE MATERIAL: {source}
Directly respond to at least one named agent's point, then decide whether to revise.
Return JSON: {{"agent":"{role}","responses":[{{"responding_to":"agent name","position":"agree|disagree|refine","reason":"...","evidence":["exact quote"]}}],"changed_position":true,"revised_recommendation":"hire|no_hire|hold","change_reason":"...","remaining_uncertainty":["..."]}}. If you do not change, set changed_position false and explain why.'''


def chair_prompt(job: str, profile: dict[str, Any], opinions: dict[str, Any], debate: dict[str, Any]) -> str:
    return f'''Act as impartial panel chair. Do not average recommendations. Weigh direct evidence, severity of role-critical risks, uncertainty, and contested claims.
JOB: {job}\nPROFILE: {json.dumps(profile, ensure_ascii=False)}\nINITIAL OPINIONS: {json.dumps(opinions, ensure_ascii=False)}\nDEBATE: {json.dumps(debate, ensure_ascii=False)}
Return JSON: {{"final_recommendation":"hire|no_hire|hold","confidence":"low|medium|high","decision_rationale":"...","strengths":[{{"claim":"...","evidence":["exact quote"]}}],"concerns":[{{"claim":"...","severity":"low|medium|high","evidence":["exact quote"]}}],"unresolved_disagreements":["..."],"information_needed":["..."]}}.'''


def main(job_file: str, resume_file: str, transcript_file: str, candidate_id: str) -> None:
    job, resume, transcript = map(pdf_text, (job_file, resume_file, transcript_file))
    sources = f"RESUME:\n{resume}\n\nTRANSCRIPT:\n{transcript}"
    profile = call_json("You are a precise evidence extractor.", profile_prompt(sources))

    # No opinion is put into these prompts: four separate, parallel API calls are independent.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {r: pool.submit(call_json, "You are an interview panel agent.", independent_prompt(r, d, job, profile)) for r, d in PERSONAS.items()}
        opinions = {role: future.result() for role, future in futures.items()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {r: pool.submit(call_json, "You are an interview panel agent.", debate_prompt(r, opinions[r], opinions, sources)) for r in PERSONAS}
        debate = {role: future.result() for role, future in futures.items()}

    final = call_json("You are the impartial chair of a hiring panel.", chair_prompt(job, profile, opinions, debate))
    report = {"candidate_id": candidate_id, "model": MODEL, "profile": profile, "independent_opinions": opinions, "debate": debate, "final_decision": final}
    invalid = evidence_errors(report, sources)
    if invalid:
        raise ValueError(f"Quotes not found in supplied documents: {invalid}")
    output = Path(__file__).parent / f"{candidate_id}_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("Usage: python interview_panel.py job.pdf resume.pdf transcript.pdf candidate_id")
    main(*sys.argv[1:])
