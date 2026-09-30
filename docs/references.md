# Research and implementation references

Reviewed 2026-09-30. Reported results belong to the cited authors; none of these
external agents has been reproduced in this workspace. Pin code revisions when
adopting an implementation and retain its license. Paper PDFs stay outside Git.

| Source | Relevance and evidence boundary |
|---|---|
| [CAGE4 source and published results](https://github.com/cage-challenge/cage-challenge-4) | Simulator and competition baseline; strong heuristics must be included. |
| [CAGE4 AI Magazine 2025](https://doi.org/10.1002/aaai.70021) | Expanded environment/competition analysis; overlaps the AAAI competition paper. |
| [Coalition Networks hierarchical MARL, MILCOM 2024](https://doi.org/10.1109/MILCOM61039.2024.10773689) | Uploaded paper; hierarchy, curriculum and communication, partly ongoing research. |
| [EPyMARL](https://github.com/uoe-agents/epymarl) | Initial MAPPO implementation candidate; adapt the actual runner interface. |
| [Hierarchical MARL for Cyber Network Defense](https://openreview.net/pdf?id=ew58AyvrlH) / [code and policies](https://github.com/adityavs14/Hierarchical-MARL) | RLJ 2025, CAGE4; independent PPO with Investigate/Recover and optional traffic policy. Not the same paper as the uploaded coalition-network study. |
| [Posterior-conditioned MARL](https://doi.org/10.1016/j.artint.2026.104606) / [Multi-Adversary-CAGE4](https://github.com/ArmitaKazemiNajafabadi/Multi-Adversary-CAGE4) | 2026 CAGE4 attacker-belief conditioning; close prior work for adaptation. Repository verified available, not executed. |
| [Entity-based defense](https://arxiv.org/abs/2410.17647) / [code](https://github.com/alan-turing-institute/Entity-Based-Yawning-Titan) | 2024, revised 2025; variable entity/topology policies demonstrated in Yawning Titan, not CAGE4. |
| [ACDZero](https://arxiv.org/abs/2601.02196) | January 2026 preprint; graph-guided MCTS/distillation in CAGE4. Later candidate; planning access and compute need review. |
| [LLM-to-lightweight policy distillation](https://arxiv.org/abs/2607.28826) | July 2026 preprint; modified CAGE2. Teacher-guided variants do not consistently beat the teacher; transfer to our MARL setting is unproven. |

The initial review deck's other bibliography entries need individual verification.
The project contribution should be a controlled result, not the claim that MAPPO,
hierarchy, memory or attacker modeling alone is new.
