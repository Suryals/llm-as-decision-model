"""The three triage decisions every router answers, defined once.

Both routers read their questions from here, so Jev and Qwen always get
identical wording. Three variants:
  plain  — one-line definitions (what Jev got in the first experiment)
  rubric — priority gets a written rubric + tie-break; the other two
           decisions are unchanged
  policy — rubric with the two policy points the error analysis found
           (upper bound: written after seeing errors on this test set)
"""

CATEGORIES = {
    "capacity": "resource capacity issue (disk, memory, CPU, quota)",
    "availability": "service outage, downtime, or failed jobs blocking users",
    "performance": "degraded latency or throughput, service still up",
    "security": "security incident, auth anomaly, or data exposure",
    "noise": "flapping, duplicate, or informational alert needing no action",
}

PRIORITIES = {
    "P1": "immediate response, customer-facing impact or SLA breach in progress",
    "P2": "urgent, needs action this shift, no customer impact yet",
    "P3": "routine, can wait for business hours",
}

# Written once, before scoring, from 18 example alerts; not tuned against results.
PRIORITY_RUBRIC = {
    "P1": "act now. Users or customers are impacted right now, OR an SLA or deadline will be breached within about an hour, OR there is a confirmed security exposure (data publicly readable, credentials leaked, access from a known-malicious source).",
    "P2": "needs a human this shift. Something is broken or degrading and will need action within hours, but no users are impacted yet and no deadline falls inside the hour. Includes suspicious-but-unconfirmed security signals, capacity approaching limits, and latency regressions without errors.",
    "P3": "can wait for business hours. Already recovered or self-healing, flapping, duplicate or informational, automation is already handling it (autoscaler replacing nodes, retries succeeding), or work that is slow but still completing with no deadline at risk.",
}
PRIORITY_TIEBREAK = (
    "If torn between two levels, pick the less urgent one unless there is "
    "evidence of impact now or a deadline within the hour."
)

# UPPER BOUND, not a fair arm: PRIORITY_RUBRIC with only the two policy
# points that the error analysis found it contradicted the labels on
# (security anomalies -> P1, slow-burn capacity/perf with days of runway -> P3).
# Written AFTER seeing the errors on this same test set, so its score is an
# upper bound on "what if the policy were written down correctly".
PRIORITY_POLICY = {
    "P1": "act now. Users or customers are impacted right now, OR an SLA or deadline will be breached within about an hour, OR there is any security anomaly that suggests compromise or data misuse, even if not yet confirmed (unusual access to sensitive data, audit logging disabled, privileged access granted unexpectedly, malware signatures, impossible-travel logins, access from a known-malicious source).",
    "P2": "needs a human this shift. Something is broken or degrading and will need action within hours, but no users are impacted yet and no deadline falls inside the hour. Includes capacity that will run out within hours and latency regressions that are getting worse.",
    "P3": "can wait for business hours. Already recovered or self-healing, flapping, duplicate or informational, automation is already handling it (autoscaler replacing nodes, retries succeeding), work that is slow but still completing with no deadline at risk, or capacity and performance trends with days of runway left.",
}

VARIANTS = ("plain", "rubric", "policy")


def questions(variant: str) -> dict[str, dict]:
    """{decision: {"type", "instructions", "criteria"?}} — the Jev request shape.
    The Qwen runner renders the same dicts into prompt text."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant in ("rubric", "policy"):
        priority = {
            "type": "choice",
            "instructions": f"What priority should this alert get? {PRIORITY_TIEBREAK}",
            "criteria": PRIORITY_RUBRIC if variant == "rubric" else PRIORITY_POLICY,
        }
    else:
        priority = {
            "type": "choice",
            "instructions": "What priority should this alert get?",
            "criteria": PRIORITIES,
        }
    return {
        "category": {
            "type": "choice",
            "instructions": "What type of alert is this?",
            "criteria": CATEGORIES,
        },
        "priority": priority,
        "page_human": {
            "type": "noul",
            "instructions": "Should this alert page an on-call human right now, "
            "as opposed to being handled by automation or a ticket?",
        },
    }
