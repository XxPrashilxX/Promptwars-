"""
Debate step.

This is the highest-weighted, most-often-rushed part of the assignment.
Each agent now sees:
  - its OWN prior opinion
  - the OTHER three agents' opinions (verdict + evidence + confidence)
and must explicitly agree, disagree, or revise — citing which other agent's
point it's responding to. We log every verdict/confidence change as a
structured diff, which is the proof artifact the rubric asks for
("show the moment an agent's opinion changed").
"""

import concurrent.futures
# pyrefly: ignore [missing-import]
import anthropic
# pyrefly: ignore [missing-import]
from agents import PERSONAS, MODEL

client = anthropic.Anthropic()

DEBATE_SCHEMA = {
    "name": "record_debate_response",
    "description": "Record this agent's response after seeing the other agents' opinions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["strong_hire", "hire", "lean_no", "no_hire", "insufficient_info"]
            },
            "confidence": {"type": "number"},
            "changed_from_initial": {
                "type": "boolean",
                "description": "True if verdict OR confidence shifted meaningfully from this agent's independent opinion"
            },
            "responses_to_others": {
                "type": "array",
                "description": "Direct responses to specific other agents' points",
                "items": {
                    "type": "object",
                    "properties": {
                        "responding_to_agent": {"type": "string", "enum": ["technical", "hr_culture", "hiring_manager", "skeptic"]},
                        "stance": {"type": "string", "enum": ["agree", "disagree", "partially_agree"]},
                        "reasoning": {"type": "string", "description": "Why, citing evidence"}
                    },
                    "required": ["responding_to_agent", "stance", "reasoning"]
                },
                "minItems": 1
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "quote_or_fact": {"type": "string"},
                        "source": {"type": "string", "enum": ["resume", "transcript"]}
                    },
                    "required": ["claim", "quote_or_fact", "source"]
                }
            }
        },
        "required": ["verdict", "confidence", "changed_from_initial", "responses_to_others", "evidence"]
    }
}


def _format_others(all_opinions: dict, exclude_key: str) -> str:
    lines = []
    for key, op in all_opinions.items():
        if key == exclude_key:
            continue
        lines.append(f"--- {op['role']} ---")
        lines.append(f"Verdict: {op['verdict']} (confidence {op['confidence']})")
        for ev in op["evidence"]:
            lines.append(f"  - {ev['claim']}: \"{ev['quote_or_fact']}\" [{ev['source']}]")
        if op.get("unknowns"):
            lines.append(f"  Unknowns: {op['unknowns']}")
        lines.append("")
    return "\n".join(lines)


def _debate_single_agent(persona_key: str, own_opinion: dict, all_opinions: dict) -> dict:
    persona = PERSONAS[persona_key]
    others_text = _format_others(all_opinions, exclude_key=persona_key)

    system = persona["system"] + """

You are now in the DEBATE phase. You will see the other three agents'
independent opinions. Do not simply restate your own opinion — you must
directly engage with at least one other agent's point: say whether you
agree, disagree, or partially agree, and why, citing evidence. If another
agent's evidence genuinely changes your assessment, update your verdict or
confidence and set changed_from_initial to true. Do not change your mind
just to seem agreeable — only change if the evidence warrants it."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system,
        tools=[DEBATE_SCHEMA],
        tool_choice={"type": "tool", "name": "record_debate_response"},
        messages=[{
            "role": "user",
            "content": f"""YOUR INDEPENDENT OPINION WAS:
Verdict: {own_opinion['verdict']} (confidence {own_opinion['confidence']})
Evidence: {own_opinion['evidence']}

THE OTHER AGENTS' INDEPENDENT OPINIONS:
{others_text}

Respond."""
        }]
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return {"agent": persona_key, "role": persona["role"], **tool_use_block.input}


def run_debate_round(all_opinions: dict) -> dict:
    """One debate round for all 4 agents, run in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_debate_single_agent, key, all_opinions[key], all_opinions): key
            for key in all_opinions
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    return results


def build_change_log(initial_opinions: dict, debate_opinions: dict, round_num: int) -> list:
    """
    The proof artifact: a structured diff of every verdict/confidence change,
    with the specific trigger (which other agent's point caused it).
    """
    changes = []
    for key, debated in debate_opinions.items():
        initial = initial_opinions[key]
        if debated["changed_from_initial"] or debated["verdict"] != initial["verdict"]:
            triggers = [
                r for r in debated["responses_to_others"]
                if r["stance"] in ("agree", "partially_agree")
            ]
            changes.append({
                "round": round_num,
                "agent": debated["role"],
                "old_verdict": initial["verdict"],
                "new_verdict": debated["verdict"],
                "old_confidence": initial["confidence"],
                "new_confidence": debated["confidence"],
                "triggered_by": triggers[0]["responding_to_agent"] if triggers else "unspecified",
                "reasoning": triggers[0]["reasoning"] if triggers else None
            })
    return changes


def run_full_debate(initial_opinions: dict, rounds: int = 2) -> dict:
    """
    Runs `rounds` of debate. Returns the final opinions plus the full change
    log across all rounds (this is what your report's "unresolved
    disagreement" section reads from).
    """
    current = initial_opinions
    all_changes = []
    for r in range(1, rounds + 1):
        next_round = run_debate_round(current)
        all_changes.extend(build_change_log(current, next_round, round_num=r))
        current = next_round  # feed forward for next round
    return {"final_opinions": current, "change_log": all_changes}
