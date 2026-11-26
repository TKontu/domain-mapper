# reconcile

Read project AGENTS.md and `.agent/config.toml`; follow [the protocol](../../SPEC.md).

Inspect actual merged/cancelled outcomes and final main. Run the configured verification and any
additional combined-state gates owed by the assignment. Record failures and their owner. Every round is reconciled and closed,
including failed, blocked and stalled ones: a failed required gate does not keep the round open.
Record it as the outcome. Resolve the assignment as merged with the evidence, or cancel it with its
next owner. Add a follow-up item for the failure. Never leave a round open for a later session to
discover; an open round blocks every later round. Confirm no workers still hold mutable resources.

Update only the canonical backlog/details: acceptance met → completed outcome (for behavior, only with an observed run or artifact reference;
merged but not yet exercised stays partial and names the run that will show it); newly discovered work
→ new item; negative measurement → evidence; target conflict → proposed decision awaiting approval.
Cancelled work remains visible with its next owner. Do not change architecture to make a result fit.
Record residuals and the actual behavior change, including none for documentation-only work.

Leave durable reconciliation evidence in tracked project state: update existing backlog/details or
decision records with outcomes, evidence, remaining work and next action. The reconciliation PR may
already capture this; do not create a separate report just to close the round. Get any required
review/merge for the record updates and retain a durable evidence reference.
## Progress review (every round)

Before the handoff, judge whether the loop is moving toward the frontier, not only whether this
round closed. Measure it against tracked state: items reaching completed or seen, milestone rows
advanced, and main changed by product work, not just records. Count consecutive rounds without
product progress from the previous handoff's `Progress:` line.

- **Advanced:** write `Progress: advanced` in the handoff.
- **Stalled (no product progress this round):** name the cause from evidence. Possible causes: a
  long lane still running, a blocked or failing item, an owed operator decision, tooling or ledger
  refusal, a permission or classifier denial, an unavailable service or model, or churn on
  low-value polish. Then redirect. Do not repeat the same plan:
  - waiting on a lane → plan other independent executable work beside it;
  - same item blocked twice → split it, change approach, or park it with its owner and pick the
    next frontier item;
  - owed to the operator → park it, open a dashboard `open` note, continue with unblocked work;
  - tooling, permission or service failure → record the cause, plan a fix item if it is in scope,
    otherwise park it and route around;
  - polish churn (3+ rounds on one area while milestone rows wait) → return to the milestone path.
  Write `Progress: stalled N (cause; redirect)` in the handoff, where N counts consecutive stalled rounds.
- **Stalled 3+ rounds with no executable work left:** write one dashboard `issue` that names the
  cause and what would unblock it. End the round with `complete --outcome continue`. Halting is
  reserved for an operator instruction or a reached goal; a stall is never a reason to halt.

After the final default-branch revision is known, run the owed final gate and write
[the handoff](handoff.md). Before it, run `python3 scripts/lanes.py cleanup --go` and carry
every held lane into the handoff with its reason. Close locally:

```sh
python3 scripts/round.py close ROUND --main main --reconciliation "<reconciliation-evidence-reference>" --handoff HANDOFF.md --evidence "Final revision check evidence; workers stopped"
```

Replace the placeholder with an existing tracked file path or an HTTP(S) evidence URL, such as the
reconciliation PR. Use the actual default branch and handoff path. No report-path configuration is
required. The CLI checks terminal assignments, merge ancestry and the handoff SHA. It checks local
evidence files at final main; it records URLs without fetching them. Verify the reference actually
shows this round's reconciliation into tracked state and that all required gates passed before
closing. Only then start a fresh round. An interrupted session may hand off this round while it
remains open; cancelled rounds also reconcile.
