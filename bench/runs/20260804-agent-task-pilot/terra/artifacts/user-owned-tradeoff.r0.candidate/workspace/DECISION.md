# Queue backend decision guide

## Status

**Decision owner:** User / product stakeholder  
**Recommendation:** Select based on the priority below; the documented hard constraints do not distinguish the options.

## Hard-constraint fit

Both **Atlas** and **Comet** meet the current requirements for:

- durability;
- regional availability;
- encryption; and
- throughput.

Either option can be migrated away from during its first month, so the initial choice is reversible within that window.

## Material trade-off

| Criterion | Atlas | Comet |
| --- | --- | --- |
| Monthly cost | Fixed at **USD 4,000** | Usage-based, **USD 2,000–7,000** depending on bursts |
| p99 latency | **42 ms** | **18 ms** |
| Operational readiness | Team already knows it | One week of operational training required |
| Reversibility | Migration reversible in first month | Migration reversible in first month |

Atlas provides a known monthly spend and immediate operational familiarity, at a measured p99 latency that is 24 ms higher than Comet's. Comet provides the lowest measured p99 latency, but monthly spend can vary by up to USD 5,000 across the documented burst range and it requires training before operations is ready.

## Conditional choice

- **Choose Atlas** if predictable monthly cost is more important than achieving the lowest available p99 latency, and avoiding a training week has value.
- **Choose Comet** if the lowest measured p99 latency is more important than fixed monthly cost, and the team accepts usage-driven spend variation plus one week of operational training.

No stated requirement establishes which of these priorities wins. The decision therefore remains with the user rather than being inferred from the technical constraints.

## Selection confirmation

Before finalizing, explicitly record one of these statements:

1. “We prioritize a fixed USD 4,000 monthly queue cost over a 24 ms p99-latency improvement; select Atlas.”
2. “We prioritize the 18 ms p99 latency over fixed monthly queue cost and accept the USD 2,000–7,000 monthly range plus training; select Comet.”

No backend implementation or configuration change is included in this decision document.
