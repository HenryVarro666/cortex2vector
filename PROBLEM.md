# Research problem

> **Can cortical-folding embeddings distinguish anatomical structure while remaining stable to representation changes and without relying on repeated-subject leakage?**

**Why it matters:** An anatomical embedding is useful for research only if measured similarity reflects the intended structure rather than file layout, mesh sampling or subject overlap.

**Central tension:** A representation may preserve individual anatomy and simultaneously make evaluation vulnerable to memorizing participants. Robustness to mesh transformations must also be tested rather than inferred from visual examples.

**Current evidence and gap:** [README.md](README.md) currently provides only a project title. [Experiment1/](Experiment1/) and [c2vExperiment1/](c2vExperiment1/) identify experiment material, but a reproducible task, baseline and split contract still need documentation. This framing does not assert that a particular embedding method or result has been verified.

**Decisive verification:** First recover the actual input, training objective and evaluation scripts. Freeze subject-disjoint partitions and compare the recovered embedding with a simple anatomical-feature baseline. Apply controlled remeshing or feature perturbations and measure neighborhood stability alongside a predeclared downstream task. If performance depends on subject overlap or representation artifacts, report that limitation before proposing model improvements.

This is a scoped research question and audit plan, not a new model claim.
