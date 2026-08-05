# Queue backend decision

## Executive summary

Both Atlas and Comet satisfy the known hard constraints: durability, region availability, encryption, and the current throughput requirement. Neither option is disqualified on those grounds. The decision therefore depends on the user's primary operating priority:

- **Choose Atlas if predictable monthly cost is the priority.** Atlas costs a fixed **USD 4,000/month**, has measured p99 latency of **42 ms**, is already familiar to the operations team, and can be migrated away from during the first month.
- **Choose Comet if the lowest latency is the priority.** Comet has measured p99 latency of **18 ms**—24 ms lower than Atlas—but its usage-based cost ranges from **USD 2,000 to 7,000/month** during bursts. It also requires one week of operational training, and migration away remains reversible during the first month.

## Comparison

| Consideration | Atlas | Comet | Decision impact |
|---|---|---|---|
| Hard constraints | Meets all stated constraints | Meets all stated constraints | Tie; no blocker |
| p99 latency | 42 ms | 18 ms | Comet is better by 24 ms |
| Monthly cost | Fixed USD 4,000 | Usage-based USD 2,000–7,000 | Atlas is more predictable; Comet may be cheaper or more expensive |
| Operations | Team already knows it | One week of training required | Atlas has lower near-term operational effort |
| Reversibility | Reversible during first month | Reversible during first month | Tie; lowers initial decision risk |

## Recommendation

There is no evidence in the stated requirements that cost predictability outranks latency, or vice versa. Ask the owner:

> **Which priority should drive the choice: predictable monthly cost, or the lowest possible p99 latency?**

Use **Atlas** for a cost-predictability-first decision. Its fixed price avoids budget surprises, and existing operational familiarity reduces rollout friction. Use **Comet** for a latency-first decision. Its measured p99 latency is substantially lower, with the trade-off of variable burst cost and a week of training.

If the priority is genuinely undecided, **Atlas is the conservative default** because its fixed cost and existing operational knowledge make the financial and operational impact easier to forecast. This is a default, not a claim that Atlas is technically faster; Comet remains the correct choice when the 24 ms p99 improvement has higher value than cost certainty.

## Scope and next step

This is a backend selection decision only. No queue backend has been implemented or migrated. Before implementation, record which priority the owner selected and, if choosing Comet, confirm that the variable cost range and training effort are acceptable. Both choices retain a reversible migration path during the first month.
