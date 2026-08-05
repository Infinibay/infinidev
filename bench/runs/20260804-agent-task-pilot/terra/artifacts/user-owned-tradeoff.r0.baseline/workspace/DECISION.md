# Queue backend decision

## Status: user decision required

Atlas and Comet both meet the known hard requirements: durability, regional availability, encryption, and current throughput. Either can be migrated away from during the first month, so the choice is reversible in that period.

The remaining decision is a trade-off between **predictable cost** and **lowest latency**:

| If the decisive priority is… | Choose | Why |
| --- | --- | --- |
| A fixed, predictable monthly spend and no new operational training | **Atlas** | It costs USD 4,000/month, the operations team already knows it, and its measured p99 latency is 42 ms. |
| The lowest measured p99 latency, accepting variable spend and training | **Comet** | Its p99 latency is 18 ms, but monthly cost can range from USD 2,000–7,000 during bursts and the team needs one week of training. |

## Decision criteria

- Select **Atlas** if a known USD 4,000 monthly cost is more valuable than reducing p99 latency by 24 ms.
- Select **Comet** if reducing p99 latency from 42 ms to 18 ms is more valuable than avoiding a USD 2,000–7,000 monthly cost range and one week of operational training.

## Next action

The decision owner should state which priority governs this workload—cost predictability or lowest latency—then approve the corresponding backend. No backend implementation or migration should begin until that choice is made.
