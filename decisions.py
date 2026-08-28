"""
Final decision step.

Explicitly NOT an average of the 4 verdicts. A fifth LLM call ("the chair")
reads the full debate transcript + change log and reasons about which
evidence is strongest, which agents' confidence was earned vs. not, and
whether disagreements were actually resolved. The rationale it produces is
what proves this isn't averaging.
"""

# pyrefly: ignore [missing-import]
import anthropic
# pyrefly: ignore [missing-import]
from agents import MODEL

client = anthropic.Anthropic()

DECISION_SCHEMA = {
    "name": "record_final_decision",
    "description": "Record the panel chair's final hiring decision after weighing all agents' evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendation": {
                "type": "string",
                "enum": ["strong_hire", "hire", "lean_no", "no_hire", "insufficient_info"]
            },
            "confidence": {"type": "number"},
            "weighting_rationale": {
                "type": "string",
                "description": "Explain WHY the decision landed here — which agent's evidence was most decisive and why, which concerns were outweighed and by what, why this is not a simple average"
            },
            "strengths": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "quote_or_fact": {"type": "string"},
                        "raised_by": {"type": "string"}
                    },
                    "required": ["point", "quote_or_fact", "raised_by"]
                }
            },
            "concerns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "quote_or_fact": {"type": "string"},
                        "raised_by": {"type": "string"}
                    },
                    "required": ["point", "quote_or_fact", "raised_by"]
                }
            },
            "unresolved_disagreements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Disagreements between agents that persisted through debate and were not settled"
            }
        },
        "required": ["recommendation", "confidence", "weighting_rationale",
                      "strengths", "concerns", "unresolved_disagreements"]
    }
}

CHAIR_SYSTEM = """You are the panel chair synthesizing a final hiring decision.
You have the full independent opinions, the debate transcript, and a log of
every opinion change. Your job is NOT to average the four verdicts. Instead:

1. Weigh evidence quality — specific, checkable evidence outweighs vague
   impressions, regardless of which agent raised it.
2. Weigh whether confidence was earned — an agent claiming high confidence
   with thin evidence should carry LESS weight than an agent with moderate
   confidence backed by strong, specific evidence.
3. Note where debate genuinely resolved a disagreement (an agent updated
   its view with good reason) vs. where it didn't (agents talked past each
   other or repeated positions).
4. If evidence is genuinely insufficient on a decisive point, don't force a
   confident verdict — say so.

Your weighting_rationale must explain your actual reasoning, not just
restate the verdicts."""


def make_final_decision(initial_opinions: dict, debate_result: dict) -> dict:
    final_opinions = debate_result["final_opinions"]
    change_log = debate_result["change_log"]

    opinions_text = "\n\n".join(
        f"--- {op['role']} (FINAL, post-debate) ---\n"
        f"Verdict: {op['verdict']} (confidence {op['confidence']})\n"
        f"Evidence: {op['evidence']}\n"
        f"Responses to others: {op.get('responses_to_others', [])}"
        for op in final_opinions.values()
    )

    changes_text = "\n".join(
        f"- {c['agent']} changed from {c['old_verdict']}->{c['new_verdict']} "
        f"(round {c['round']}), triggered by {c['triggered_by']}: {c['reasoning']}"
        for c in change_log
    ) or "No agent changed its verdict during debate."

    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=CHAIR_SYSTEM,
        tools=[DECISION_SCHEMA],
        tool_choice={"type": "tool", "name": "record_final_decision"},
        messages=[{
            "role": "user",
            "content": f"""FINAL AGENT OPINIONS (post-debate):
{opinions_text}

CHANGE LOG FROM DEBATE:
{changes_text}

Make the final decision."""
        }]
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return tool_use_block.input
