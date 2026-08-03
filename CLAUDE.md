@../../CLAUDE.md

## Performance / batch runners

fenicsx is the production backend. For multi-job sweeps (invsec etc.) follow
**[SKILL.md](SKILL.md)** — warm workers, `linear_solver=lu`, one spawn pool,
shared FFCx cache. Numbers in `notes/mind/speed.md`.
