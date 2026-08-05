# Queue backend decision

## Decision context

Both Atlas and Comet satisfy the known hard constraints:

- Durability
- Region availability
- Encryption
- Current throughput

The remaining decision is a business priority that has not been specified: choose predictable monthly cost, or choose the lowest measured latency.

## Option comparison

| Criterion | Atlas | Comet |
|---|---|---|
| Monthly cost | **$4,000 fixed** | **$2,000–$7,000**, depending on bursts |
| Measured p99 latency | 42 ms | **18 ms** |
| Operational familiarity | Operations team already knows it | Requires one week of operational training |
| Migration reversibility | Reversible during the first month | Reversible during the first month |
| Hard constraints | Satisfies all known constraints | Satisfies all known constraints |

## Recommendation structure

### Choose Atlas if cost predictability is the priority

Atlas is the better fit when the budget must be forecastable and spend variance is more important than minimizing tail latency. Its fixed $4,000/month price and existing operational familiarity reduce planning and adoption risk. The trade-off is a measured p99 latency of 42 ms.

### Choose Comet if lowest latency is the priority

Comet is the better fit when tail-latency performance matters most. Its measured p99 latency of 18 ms is 24 ms lower than Atlas’s (about 57% lower), but its monthly cost can range from $2,000 to $7,000 depending on burst usage. The choice also requires one week of operational training. Migration remains reversible during the first month.

## Decision requested

Which priority should govern the choice?

Select the priority that should govern the choice:

- **Predictable cost → Atlas**
- **Lowest latency → Comet**

There is no constraint in the current requirements that resolves this trade-off. Until that priority is selected, neither option is objectively preferred; both remain technically viable and reversible during the first month.

## Scope

This document records the backend choice criteria only. It does not implement, provision, or migrate either queue backend.
