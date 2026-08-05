# Queue backend decision

Both options satisfy durability, region availability, encryption, and current throughput constraints.

## Atlas

- Predictable fixed cost: USD 4,000/month.
- p99 latency measured at 42 ms.
- Operations already know it.
- Reversible migration during the first month.

## Comet

- Usage-based cost: USD 2,000–7,000/month depending on bursts.
- p99 latency measured at 18 ms.
- Requires one week of operational training.
- Reversible migration during the first month.

No stated requirement prioritizes cost predictability over latency or latency over cost predictability.
