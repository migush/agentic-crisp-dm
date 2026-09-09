# CRISP-DM Loop Contours

Fire back-edges only when triggers are present in state:

- **Loop A (2→1):** Fire only when **actionable** quality issues remain after considering `quality_gate` (`na_means_absent`, `high_missing`, `domain_data_quality_flags`, `loop_a_recommendation`) — not when blockers are only structural absence already documented. Return to 1.3.
- **Loop B (4→3):** `validator_findings`, `degraded_flags`, or CV below threshold → return to Phase 3 (max 3×). Decided at checkpoint 5.1 **before** Evaluate Results.
- **Loop C (5.2→1):** after Evaluate Results, if business success criteria not met → return to 1.3 (at most once). Halt suggested if Loop A fired twice. Do not fire C at 5.1 — that checkpoint has not re-evaluated this visit.
- **Loop D (6→1):** optional after 6.4; experience feeds next run knowledge

When a loop is blocked by visit/inner caps, **advance** the current phase instead of halting so a submission can still be produced.

Output directive JSON: `action` (advance|loop_back|halt), `loop_label`, `loop_to_phase`, `target_substep`, `reason`.
