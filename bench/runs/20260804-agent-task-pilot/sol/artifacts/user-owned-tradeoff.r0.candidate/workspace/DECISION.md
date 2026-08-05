# Queue backend decision

## Decision status

No backend is selected yet. Atlas and Comet both satisfy the required durability, regional availability, encryption, and current throughput constraints, and migration is reversible during the first month. The deciding priority must therefore come from the user: **cost predictability** or **lowest measured latency**.

## Evidence

| Option | Monthly cost | Measured p99 latency | Operational impact |
| --- | --- | --- | --- |
| Atlas | Fixed USD 4,000 | 42 ms | Already known by operations |
| Comet | Usage-based USD 2,000–7,000, depending on bursts | 18 ms | Requires one week of operational training |

## Decision rule

- Choose **Atlas** if predictable monthly cost is the higher priority.
- Choose **Comet** if the lowest measured p99 latency is the higher priority and variable monthly cost plus the training requirement are acceptable.

Neither priority is established by the current requirements, so the evidence does not support an unconditional recommendation. Which priority should decide the selection: predictable monthly cost or the lowest measured p99 latency?
