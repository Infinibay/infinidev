# Queue backend decision

## Decision status

Both Atlas and Comet meet the known hard requirements for durability, region availability, encryption, and current throughput. The final choice therefore depends on which priority matters more to the user: predictable monthly cost or the lowest measured p99 latency.

## Known facts

| Option | Cost | Measured p99 latency | Operational impact | Migration |
| --- | --- | --- | --- | --- |
| Atlas | Fixed USD 4,000/month | 42 ms | Already known by operations | Reversible during the first month |
| Comet | USD 2,000–7,000/month, depending on bursts | 18 ms | Requires one week of operational training | Reversible during the first month |

## Conditional recommendation

- **Choose Atlas if cost predictability is the decisive priority.** It fixes monthly spend at USD 4,000 and requires no new operational training, in exchange for a higher measured p99 latency than Comet.
- **Choose Comet if the lowest measured p99 latency is the decisive priority.** Its measured p99 latency is 18 ms rather than Atlas's 42 ms, in exchange for variable monthly spend and one week of operational training.

## Assumptions and unresolved priority

No requirement states that cost predictability is more important than latency, or vice versa. This document therefore does not select an unconditional winner. It also does not assume that Comet's variable cost will settle at a particular point within its stated range or that either latency result has a specific business impact beyond the measurements provided.

The user should select the applicable branch above based on the priority they are willing to optimize. Either selection remains reversible during the first month.
