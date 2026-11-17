from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("ollama:qwen3")  # anthropic:claude-haiku-4-5

# US-style phone numbers: 555-201-9988, 555.201.9988, 5552019988, etc.
PHONE_PATTERN = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"

BASELINE_REFUND_PROMPT = (
    "You help resolve billing complaints. Read the customer's raw "
    "message, and if a refund is clearly warranted, call issue_refund "
    "with their actual email, a specific amount, and a one-line "
    "reason. If a refund isn't clearly warranted, explain why instead "
    "of calling the tool."
)

REVISE_REFUND_PROMPT = (
    BASELINE_REFUND_PROMPT
    + " If a human rejects your issue_refund call and gives a reason, "
    "either call issue_refund again with arguments that satisfy that "
    "reason, or explain why no refund can be issued and do not call "
    "the tool. Do not ignore the rejection reason."
)

sanitize_intake = create_agent(
    model,
    tools=[],
    system_prompt=(
        "Summarize this support message in 2-3 sentences: what the "
        "customer needs, and how urgent it sounds. Do not repeat any "
        "contact details back in your summary."
    ),
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware(
            "phone",
            detector=PHONE_PATTERN,
            strategy="redact",
            apply_to_input=True,
        ),
    ],
)


@tool
def issue_refund(customer_email: str, amount: str, reason: str) -> str:
    """Issue a refund to a customer. This actually issues it -- treat every call as real."""
    return f"Refund of {amount} issued to {customer_email} ({reason})."


def make_draft_refund_agent(system_prompt: str):
    # No checkpointer on this inner agent -- only the outer graph is
    # checkpointed, so it's the one that catches the interrupt. See
    # middleware_example_hitl.py's docstring for why that matters.
    return create_agent(
        model,
        tools=[issue_refund],
        system_prompt=system_prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "issue_refund": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    },
                },
                description_prefix="This refund is about to be issued for real:",
            ),
        ],
    )


class SupportState(TypedDict):
    raw_message: str
    redacted_summary: str
    category: str
    refund_outcome: str
    revision_count: int
    last_rejection_reason: str


def sanitize_intake_node(state: SupportState) -> dict:
    result = sanitize_intake.invoke(
        {"messages": [{"role": "user", "content": state["raw_message"]}]}
    )
    return {"redacted_summary": result["messages"][-1].text}


def triage_node(state: SupportState) -> dict:
    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Classify this support summary into exactly one category: "
                    "BILLING, TECHNICAL, or ACCOUNT. Reply with just the word."
                ),
            },
            {"role": "user", "content": state["redacted_summary"]},
        ]
    )
    return {"category": response.text.strip().upper()}


def route_after_triage(state: SupportState) -> str:
    # Only a billing ticket has any business reaching an agent whose
    # entire system prompt is about resolving billing complaints.
    if state["category"] == "BILLING":
        return "draft_refund_action"
    return END


def build_support_graph(refund_system_prompt: str):
    draft_refund_action = make_draft_refund_agent(refund_system_prompt)

    def draft_refund_action_node(state: SupportState) -> dict:
        # Deliberately reads raw_message, not redacted_summary -- see docstring.
        result = draft_refund_action.invoke(
            {"messages": [{"role": "user", "content": state["raw_message"]}]}
        )
        return {"refund_outcome": result["messages"][-1].text}

    graph = StateGraph(SupportState)
    graph.add_node("sanitize_intake", sanitize_intake_node)
    graph.add_node("triage", triage_node)
    graph.add_node("draft_refund_action", draft_refund_action_node)
    graph.add_edge(START, "sanitize_intake")
    graph.add_edge("sanitize_intake", "triage")
    graph.add_conditional_edges(
        "triage", route_after_triage, ["draft_refund_action", END]
    )
    graph.add_edge("draft_refund_action", END)
    return graph.compile(checkpointer=InMemorySaver())


# Default demo graph uses the reject→revise prompt.
support_graph = build_support_graph(REVISE_REFUND_PROMPT)
# support_graph.get_graph().draw_mermaid_png(output_file_path="support_workflow.png")

DecisionFn = Callable[[list | dict, int], dict]


def _interactive_decide(actions: list | dict, revision_count: int) -> dict:
    print("\n--- Needs human approval ---")
    if revision_count:
        print(f"(revision round after {revision_count} prior rejection(s))")
    print(actions)
    decision_type = input("approve / edit / reject? ").strip().lower()

    if decision_type == "approve":
        return {"type": "approve"}
    if decision_type == "reject":
        reason = input("Reason: ")
        return {"type": "reject", "message": reason}

    new_amount = input("Corrected amount: ")
    return {
        "type": "edit",
        "edited_action": {
            "name": "issue_refund",
            "args": {"amount": new_amount},
        },
    }


def _action_requests(interrupt_value: Any) -> list:
    if isinstance(interrupt_value, dict):
        requests = interrupt_value.get("action_requests", interrupt_value)
        if isinstance(requests, list):
            return requests
        return [requests]
    return [interrupt_value]


def _proposed_amount(actions: list | dict) -> str:
    requests = actions if isinstance(actions, list) else [actions]
    for req in requests:
        if not isinstance(req, dict):
            continue
        args = req.get("args") or req.get("action", {}).get("args") or {}
        if "amount" in args:
            return str(args["amount"])
        name = req.get("name") or req.get("action", {}).get("name")
        if name == "issue_refund" and isinstance(req.get("args"), dict):
            return str(req["args"].get("amount", ""))
    return ""


def run_with_approval(
    initial_state: dict,
    thread_id: str,
    *,
    graph=None,
    decide: DecisionFn | None = None,
    max_revisions: int = 2,
    max_interrupts: int = 6,
) -> dict:
    """
    Invoke, check for a pause, collect a human decision, resume,
    repeat until the graph actually finishes.

    On reject, the human reason is returned to the refund agent as a
    ToolMessage (HITL middleware). After ``max_revisions`` rejects, the
    resume message also tells the model to stop calling issue_refund.
    """
    active_graph = graph if graph is not None else support_graph
    decide_fn = decide or _interactive_decide
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "revision_count": 0,
        "last_rejection_reason": "",
        **initial_state,
    }
    result = active_graph.invoke(state, config=config)
    revision_count = int(state.get("revision_count") or 0)
    last_rejection_reason = str(state.get("last_rejection_reason") or "")
    proposed_amounts: list[str] = []
    interrupt_n = 0

    while "__interrupt__" in result:
        interrupt_n += 1
        if interrupt_n > max_interrupts:
            raise RuntimeError(
                f"Exceeded max_interrupts={max_interrupts} on thread {thread_id}"
            )

        interrupt = result["__interrupt__"][0]
        actions = _action_requests(interrupt.value)
        amount = _proposed_amount(actions)
        if amount:
            proposed_amounts.append(amount)

        decision = decide_fn(actions, revision_count)

        if decision.get("type") == "reject":
            revision_count += 1
            reason = str(decision.get("message") or "").strip() or (
                "Rejected without a reason."
            )
            last_rejection_reason = reason
            if revision_count >= max_revisions:
                decision = {
                    "type": "reject",
                    "message": (
                        f"{reason}\n\nFinal rejection after {revision_count} "
                        "attempt(s). Do not call issue_refund again. Explain "
                        "that the refund was not issued."
                    ),
                }
            else:
                decision = {"type": "reject", "message": reason}

            active_graph.update_state(
                config,
                {
                    "revision_count": revision_count,
                    "last_rejection_reason": last_rejection_reason,
                },
            )

        result = active_graph.invoke(
            Command(resume={"decisions": [decision]}), config=config
        )

    result = dict(result)
    result["revision_count"] = revision_count
    result["last_rejection_reason"] = last_rejection_reason
    result["proposed_amounts"] = proposed_amounts
    return result


SAMPLE_MESSAGES = {
    "billing_refund_warranted": (
        "Hi, this is Dana Price. My card ending in 4319 2100 2222 3344 was "
        "charged twice this month -- $49.99 twice on the same day. Call me "
        "back at 555-201-9988 or email dana.price@example.com if you need "
        "anything else -- this is urgent, I need a refund before Friday."
    ),
    "billing_refund_not_warranted": (
        "This is Marcus Yi. I just noticed my plan went from $19.99 to "
        "$24.99 this month and I want a refund for the difference. I know "
        "you sent an email about the price change back in September but I "
        "don't think that's fair. You can reach me at "
        "marcus.yi88@example.com."
    ),
    "technical_password_reset": (
        "Hey, I'm locked out of my account -- the reset link you emailed "
        "isn't working and I've tried three times. My number is "
        "555-884-2201 if someone can just call me back today."
    ),
    "account_update_request": (
        "I need to update the mailing address on file for my account and "
        "add my spouse as an authorized user. The new address isn't "
        "saving in the portal. You can email me at r.chen.acct@example.com "
        "to confirm once it's done."
    ),
    "misplaced_off_topic": (
        "Hi there! I saw your company is hiring for a customer success "
        "role on LinkedIn -- is this still open, and who should I send my "
        "resume to? Thanks!"
    ),
}


def _initial_state(raw_message: str) -> dict:
    return {
        "raw_message": raw_message,
        "redacted_summary": "",
        "category": "",
        "refund_outcome": "",
        "revision_count": 0,
        "last_rejection_reason": "",
    }


def _amount_looks_like_single_charge(amount: str) -> bool:
    """True if amount text looks like one $49.99 charge, not a double."""
    if not amount:
        return False
    digits = re.findall(r"\d+(?:\.\d+)?", amount.replace(",", ""))
    if not digits:
        return False
    values = [float(x) for x in digits]
    # Accept 49.99 / 50-ish single refund; reject obvious doubles (~99.98).
    if any(90 <= v <= 110 for v in values):
        return False
    return any(45 <= v <= 55 for v in values)


def _scripted_reject_then_approve(reject_reason: str) -> DecisionFn:
    step = {"n": 0}

    def decide(actions: list | dict, revision_count: int) -> dict:
        step["n"] += 1
        print("\n--- Needs human approval (scripted) ---")
        print(f"interrupt #{step['n']} revision_count={revision_count}")
        print(actions)
        if step["n"] == 1:
            print(f"scripted decision: reject -- {reject_reason}")
            return {"type": "reject", "message": reject_reason}
        print("scripted decision: approve")
        return {"type": "approve"}

    return decide


def _scripted_reject_only(reject_reason: str) -> DecisionFn:
    def decide(actions: list | dict, revision_count: int) -> dict:
        print("\n--- Needs human approval (scripted) ---")
        print(f"revision_count={revision_count}")
        print(actions)
        print(f"scripted decision: reject -- {reject_reason}")
        return {"type": "reject", "message": reject_reason}

    return decide


def _scripted_approve() -> DecisionFn:
    def decide(actions: list | dict, revision_count: int) -> dict:
        print("\n--- Needs human approval (scripted) ---")
        print(actions)
        print("scripted decision: approve")
        return {"type": "approve"}

    return decide


def run_eval_reject_revise() -> None:
    """
    Before/after scripted eval for reject→revise.

    Prints a compact table. Uses Ollama; expect several minutes.
    """
    revise_reason = (
        "Only refund one of the duplicate charges ($49.99), not both."
    )
    no_refund_reason = (
        "Do not issue any refund; the customer is not entitled to one."
    )
    vague_reason = "no"
    zero_reason = "refund $0"

    modes = [
        ("baseline", BASELINE_REFUND_PROMPT, 99),
        ("revised", REVISE_REFUND_PROMPT, 2),
    ]
    rows: list[dict[str, Any]] = []

    for mode_name, prompt, max_revisions in modes:
        graph = build_support_graph(prompt)

        # Case 1: reject with concrete fix, then approve a re-draft if any.
        final = run_with_approval(
            _initial_state(SAMPLE_MESSAGES["billing_refund_warranted"]),
            thread_id=f"eval-{mode_name}-happy-revise",
            graph=graph,
            decide=_scripted_reject_then_approve(revise_reason),
            max_revisions=max_revisions,
        )
        amounts = final.get("proposed_amounts") or []
        second = amounts[1] if len(amounts) > 1 else ""
        rows.append(
            {
                "mode": mode_name,
                "case": "happy_revise",
                "interrupts": len(amounts),
                "second_amount": second or "(none)",
                "second_ok": _amount_looks_like_single_charge(second),
                "revision_count": final.get("revision_count", 0),
                "category": final.get("category", ""),
            }
        )

        # Case 2: reject means no refund at all.
        final = run_with_approval(
            _initial_state(SAMPLE_MESSAGES["billing_refund_warranted"]),
            thread_id=f"eval-{mode_name}-no-refund",
            graph=graph,
            decide=_scripted_reject_only(no_refund_reason),
            max_revisions=max_revisions,
        )
        amounts = final.get("proposed_amounts") or []
        rows.append(
            {
                "mode": mode_name,
                "case": "reject_no_refund",
                "interrupts": len(amounts),
                "second_amount": amounts[1] if len(amounts) > 1 else "(none)",
                "second_ok": len(amounts) <= 1,
                "revision_count": final.get("revision_count", 0),
                "category": final.get("category", ""),
            }
        )

        # Case 3: non-billing must not interrupt.
        final = run_with_approval(
            _initial_state(SAMPLE_MESSAGES["technical_password_reset"]),
            thread_id=f"eval-{mode_name}-technical",
            graph=graph,
            decide=_scripted_approve(),
            max_revisions=max_revisions,
        )
        amounts = final.get("proposed_amounts") or []
        rows.append(
            {
                "mode": mode_name,
                "case": "technical_no_interrupt",
                "interrupts": len(amounts),
                "second_amount": "(n/a)",
                "second_ok": len(amounts) == 0,
                "revision_count": final.get("revision_count", 0),
                "category": final.get("category", ""),
            }
        )

    # Adversarial cases on the revised prompt only.
    graph = build_support_graph(REVISE_REFUND_PROMPT)
    for case_name, reason in [
        ("vague_reject", vague_reason),
        ("zero_refund_reject", zero_reason),
    ]:
        final = run_with_approval(
            _initial_state(SAMPLE_MESSAGES["billing_refund_warranted"]),
            thread_id=f"eval-adv-{case_name}",
            graph=graph,
            decide=_scripted_reject_then_approve(reason),
            max_revisions=2,
        )
        amounts = final.get("proposed_amounts") or []
        rows.append(
            {
                "mode": "revised",
                "case": case_name,
                "interrupts": len(amounts),
                "second_amount": amounts[1] if len(amounts) > 1 else "(none)",
                "second_ok": len(amounts) > 1,
                "revision_count": final.get("revision_count", 0),
                "category": final.get("category", ""),
            }
        )

    # Approve-path smoke on revised prompt.
    final = run_with_approval(
        _initial_state(SAMPLE_MESSAGES["billing_refund_warranted"]),
        thread_id="eval-revised-approve-smoke",
        graph=graph,
        decide=_scripted_approve(),
        max_revisions=2,
    )
    amounts = final.get("proposed_amounts") or []
    rows.append(
        {
            "mode": "revised",
            "case": "approve_smoke",
            "interrupts": len(amounts),
            "second_amount": "(n/a)",
            "second_ok": len(amounts) == 1
            and "Refund" in (final.get("refund_outcome") or ""),
            "revision_count": final.get("revision_count", 0),
            "category": final.get("category", ""),
        }
    )

    print("\n" + "=" * 70)
    print("reject→revise eval")
    print("=" * 70)
    print(
        f"{'mode':<10} {'case':<24} {'intr':>4} {'rev':>3} "
        f"{'ok':>3}  second_amount / notes"
    )
    for row in rows:
        ok = "Y" if row["second_ok"] else "N"
        print(
            f"{row['mode']:<10} {row['case']:<24} {row['interrupts']:>4} "
            f"{row['revision_count']:>3} {ok:>3}  {row['second_amount']}"
        )


def run_demo() -> None:
    i = 1
    for label, raw_message in SAMPLE_MESSAGES.items():
        print(f"\n{'=' * 70}\nRunning message {i}: {label}\n{'=' * 70}")
        final = run_with_approval(
            _initial_state(raw_message),
            thread_id=label,
        )
        print("\n--- Redacted summary (this is all `triage` ever saw) ---")
        print(final["redacted_summary"])
        print("\n--- Category ---")
        print(final["category"])
        print("\n--- Revision count ---")
        print(final.get("revision_count", 0))
        if final.get("last_rejection_reason"):
            print("\n--- Last rejection reason ---")
            print(final["last_rejection_reason"])
        print("\n--- Refund outcome ---")
        print(
            final["refund_outcome"]
            or "(skipped -- not a BILLING ticket, or no refund action taken)"
        )
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 6 HITL support graph")
    parser.add_argument(
        "--eval-reject-revise",
        action="store_true",
        help="Run scripted before/after reject→revise eval (uses Ollama)",
    )
    args = parser.parse_args()
    if args.eval_reject_revise:
        run_eval_reject_revise()
    else:
        run_demo()
