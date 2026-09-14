# Aggregation pipeline

`pipeline(start)` pushes `start` through the whole hierarchy in order:

    8 hubs  ->  5 packages per hub  ->  30 leaves per package

Every leaf applies one fixed positive offset of its own; the packages and the
hubs only chain them. Nothing else in the repository modifies the value, so

    pipeline(0) == the sum of every leaf's offset

which this checkout computes to **14398**.

Leaves live in `src/pkgs/pkg_NN/mod_MM.py`; each package's `apply` chains its
thirty leaves, each hub's `apply` chains its five packages, and
`src/pipeline.py` chains the eight hubs. The hierarchy is the intended way to
navigate: the leaves are deliberately uniform, so searching the tree by hand is
not the cheap path.
