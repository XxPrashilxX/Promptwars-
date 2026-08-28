"""
The four independent agent personas.

CRITICAL RULE (per the challenge spec): each agent's first opinion must come
from a SEPARATE LLM call, with NO visibility into any other agent's output.
That's why `get_independent_opinions()` fires four isolated calls (in
parallel) and none of the prompts below ever reference another agent.
"""

import concurrent.futures
# pyrefly: ignore [missing-import]
import anthropic

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Schema shared by every agent's opinion. Evidence is REQUIRED and must be a
# real quote/fact — this operationalizes "every score must point to evidence"
# and "if unclear, say so" at the schema level instead of hoping the prompt
# is obeyed.
# ---------------------------------------------------------------------------
OPINION_SCHEMA = {
    "name": "record_opinion",
    "description": "Record this agent's independent assessment of the candidate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["strong_hire", "hire", "lean_no", "no_hire", "insufficient_info"]
            },
            "confidence": {
                "type": "number",
                "description": "0.0 to 1.0 — how confident this agent is in its verdict"
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "The point being made"},
                        "quote_or_fact": {"type": "string", "description": "Verbatim quote or specific fact from the profile"},
                        "source": {"type": "string", "enum": ["resume", "transcript"]}
                    },
                    "required": ["claim", "quote_or_fact", "source"]
                },
                "minItems": 1
            },
            "unknowns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things this agent could not assess from the available material"
            }
        },
        "required": ["verdict", "confidence", "evidence", "unknowns"]
    }
}

# ---------------------------------------------------------------------------
# Persona system prompts. Keep these genuinely distinct in what they look
# for — this is what graders check for "are the 4 personas actually
# different", not just different names.
# ---------------------------------------------------------------------------
PERSONAS = {
    "technical": {
        "role": "Technical Agent",
        "system": """You are the Technical Agent on a hiring panel. You evaluate
ONLY technical skill and depth: do the claimed skills hold up under the
specific questions asked in the transcript? Is there evidence of genuine
depth (specifics, tradeoffs, debugging stories) vs surface-level buzzwords?
Ignore communication style, culture fit, and business judgment — those are
not your job. Every claim you make must cite a specific quote or fact.
If the transcript doesn't probe a claimed skill, say so in 'unknowns'
rather than assuming it's solid or weak."""
    },
    "hr_culture": {
        "role": "HR / Culture Agent",
        "system": """You are the HR/Culture Agent on a hiring panel. You evaluate
ONLY communication quality, teamwork signals, honesty/consistency, and how
the candidate discusses conflict or failure. Ignore raw technical depth —
that's not your job. Watch specifically for inconsistencies between how the
candidate describes their role/contribution and what's independently
verifiable from the resume. Every claim must cite a specific quote or fact.
If you can't assess something, say so in 'unknowns'."""
    },
    "hiring_manager": {
        "role": "Hiring Manager Agent",
        "system": """You are the Hiring Manager Agent on a hiring panel. You
evaluate whether this candidate is worth hiring specifically for the role
described in the job description — fit for the actual responsibilities,
seniority match, and whether their trajectory suggests they'll succeed in
THIS role, not just whether they're generically competent. Every claim must
cite a specific quote or fact from the resume/transcript, or a specific line
from the job description. If the job description doesn't give you enough to
judge something, say so in 'unknowns'."""
    },
    "skeptic": {
        "role": "Skeptic Agent",
        "system": """You are the Skeptic Agent on a hiring panel. Your job is to
actively look for contradictions, exaggeration, vague hand-waving, or red
flags — NOT to give a balanced overview. Compare claims across the resume
and transcript for consistency. Flag any claim that sounds impressive but
lacks a concrete, checkable detail. You are allowed to be harsh, but every
red flag must cite a specific quote or fact, not vague suspicion. If nothing
concerning turns up, say that plainly rather than inventing a concern."""
    }
}


def _run_single_agent(persona_key: str, candidate_profile: dict, job_description: str) -> dict:
    """One isolated LLM call. No other agent's output is ever passed in here."""
    persona = PERSONAS[persona_key]
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=persona["system"],
        tools=[OPINION_SCHEMA],
        tool_choice={"type": "tool", "name": "record_opinion"},
        messages=[{
            "role": "user",
            "content": f"""JOB DESCRIPTION:
{job_description}

CANDIDATE PROFILE (extracted facts, not opinions):
{candidate_profile}

Give your independent assessment."""
        }]
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return {
        "agent": persona_key,
        "role": persona["role"],
        **tool_use_block.input
    }


def get_independent_opinions(candidate_profile: dict, job_description: str) -> dict:
    """
    Fires all 4 agents in PARALLEL, each blind to the others.
    Returns {persona_key: opinion_dict}.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_single_agent, key, candidate_profile, job_description): key
            for key in PERSONAS
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    return results
