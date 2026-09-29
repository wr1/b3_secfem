# b3_secfem

fenicsx is the production backend. For multi-job sweeps (invsec etc.) follow
**[SKILL.md](SKILL.md)** — warm workers, `linear_solver=lu`, one spawn pool,
shared FFCx cache. Measured warm-solve numbers are in that file.

Monorepo agents: see `b3_master/CLAUDE.md`.
