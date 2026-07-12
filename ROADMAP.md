# ElephantFormer Roadmap

**From "loses to random" to the strongest open neural xiangqi system.**

## Mission & honest framing

North star: surpass Pikafish. Reality check, stated once so the plan stays honest:

- Distilling from any teacher caps below the teacher. Self-play RL is the only method with no ceiling.
- Pikafish examines ~10⁷ exact positions per move; a forward pass examines one. Params buy the *head* of the position distribution (patterns), search nodes buy the *tail* (exact calculation). Every frontier engine ships both.
- Nobody has dethroned a Stockfish-family engine solo. Passing full Pikafish is a community-project outcome (the "Lc0 of xiangqi" path). Everything below is sequenced so that each phase has standalone value even if the project stops there.

**Headline metric:** Elo vs Pikafish restricted to N nodes. One chart: N on the x-axis, our Elo relative to Pikafish@N on the y-axis, one curve per model generation. Climbing N is the whole game.

## Current state (updated 2026-07-12)

- **Engine legality fixed** (stale-board attack test in `is_square_attacked_by`; mate-avoidance heuristic removed from `get_all_legal_moves`). All pre-fix results — the old 46.2% win rate, the README's sequence-length findings — are unreliable. Skip rate on real games went 63% → 0.05%.
- **Phase 0 model trained** (WXF corpus, 3.57M positions, 10 epochs on a single consumer GPU): test policy accuracy **48.8%** (legacy model: 12.49%), value accuracy 59.8%.
- **Scoreboard**: vs greedy-material **76%** (bar 60% — passed) · vs random **76%** (bar 95% — not yet; identical score vs both opponents ⇒ losses are searchless mate-blunders, not opponent strength).
- **Assets**: content-keyed position caches (`data/cache/`: WXF 3.57M + dpxq 8.11M ≈ 11.7M positions), Pikafish + resumable/shardable annotator (`pikafish_annotator.py`, ~40 pos/s/core measured), resume + step-checkpoint + multi-PGN training, job monitor `scripts/training_status.py`, detached jobs logging to `logs/`.
- **In flight**: combined 11.7M-position training run → `checkpoints/board-combined/`.
- **Next**: value-head 1-ply rerank bot (attack the blunder rate) → rerun scoreboard; overnight ~1M-position Pikafish annotation; game-level splits.
- Legacy assets still in use: rules engine (`elephant_former/engine/`), PGN/ICCS parser, Lightning conventions.

## Principles (what the 2024–2026 literature says)

1. **Throughput first.** Every success story (KataGo, Pgx, Generals.io 2026) reduces to simulator speed removing the data bottleneck.
2. **Board state in, move history out.** Validated by DeepMind searchless chess (2024) and Tencent searchless xiangqi (2024).
3. **Legal-move input channels for both sides** — Tencent's cheapest quantified win (~30% faster to milestones, better endgames).
4. **Transformer over CNN is confirmed for xiangqi** (Tencent: ~50% better endgame accuracy at equal params).
5. **Distill first (cheap Elo from a free teacher), self-play later (unbounded).**
6. **Measure with SPRT or don't claim the Elo.**

---

## Phase 0 — Make the model see

**Goal:** fix the representation. The single highest-leverage change in the project.
**Exit criteria:** ≥95% wins vs random · ≥60% vs greedy-material bot · ≥30% move-match on held-out human games.
**Effort:** ~2–4 weekends. Current GPU is fine.

- [x] Board-state input: 90 squares as tokens (piece-type embedding per square) + side-to-move token
- [x] Legal-move feature channels for **both** sides (from the existing engine — a day of work)
- [x] Replace the 4 independent coordinate heads with a **from-square → to-square policy head** (masked to legal moves); attention-based source–destination head à la Chessformer
- [x] Add a **value head** (win/draw/loss) as a multi-task loss
- [x] Dense loss — one example per position (the O(n²) prefix scheme is gone with the representation change)
- [ ] Replace learned absolute positions with **geometric attention bias** (file/rank/palace/river relations) — deferred: the board is fixed 90 squares, so learned rank+file embeddings suffice for now
- [x] New baseline opponent: greedy material bot alongside the random bot (greedy beats random 8/10 — sanity confirmed)
- [x] Keep: engine, parser, dataset plumbing, Lightning loop

**Status 2026-07-12:** exit criteria — move-match 48.8% ✓ (bar 30%) · vs greedy 76% ✓ (bar 60%) · vs random 76% ✗ (bar 95%). Remaining gap is searchless blunders; being attacked via more data (combined 11.7M run in flight) and a value-head 1-ply rerank at inference.

Architecture target: encoder-only, ~91-token input, 8–12 layers, d_model 256–384 (≈10–30M params).

## Phase 1 — Distillation + search

**Goal:** real strength from free teachers.
**Exit criteria:** SPRT parity with **Pikafish@1k nodes** · playable web demo.
**Effort:** ~2–3 months part-time. Annotation runs on CPU overnight.

- [x] Pikafish annotation pipeline: UCI wrapper, MultiPV top-k (best move + eval) labels — built, tested (incl. live engine), resumable, shardable across machines (`--start/--end` + `--merge`); ~40 pos/s/core at 10k nodes. Target 1–5M positions: pending the big run
- [ ] Mixed training data: human moves (breadth/style) + engine labels (accuracy) — Tencent used roughly 1:2
- [ ] MCTS at inference: batched leaf evaluation, **Gumbel root action selection**, 100–800 sims
- [ ] Eval harness v2 (spec below): Pikafish node-ladder + SPRT match runner
- [ ] Web demo (already on the README TODO — this is when it becomes worth building)

## Phase 2 — The strategic investment: throughput

**Goal:** self-play data generation stops being the bottleneck. Gates all RL.
**Exit criteria:** ≥10⁵ moves/sec on one GPU (JAX lane) or ≥10⁴ games/hour (C++ lane) · perft suite passes vs both the Python engine and pyffish.
**Effort:** ~1–3 months. The hardest engineering in the project.

- [ ] Choose lane:
  - **JAX vectorized env** (Pgx-style batched movegen; 10–100× throughput; doubles as an open-source contribution — Pgx has no xiangqi)
  - **C++ engine + Python bindings** (pyffish/Fairy-Stockfish as reference implementation and intermediate speedup)
- [ ] Perft-style test suite; cross-validate thousands of games against the existing Python engine
- [ ] Perpetual check/chase adjudication strategy: exact rules where cheap, conservative draw adjudication + engine-verified disputes elsewhere; log and audit
- [ ] Self-play worker: replay buffer, batched inference, resign threshold with false-positive audits

**Risk callout:** chase-rule bugs become RL reward hacks. No RL until the rule test suite passes.

## Phase 3 — Reinforcement learning

**Goal:** pass the teacher.
**Exit criteria:** +300 Elo over the Phase-1 model · parity with **Pikafish@10k–100k nodes**.
**Effort:** 6–12 months part-time; scales with GPUs (1–8).

Two validated lanes sharing ~90% of infrastructure:

| | Lane A — searchless (Tencent recipe) | Lane B — search (AlphaZero lane) |
|---|---|---|
| Algorithm | **VECT-PPO** (vanilla GAE collapsed in their ablation: 10.2% vs 92.5%) | **Gumbel AlphaZero** at 16–64 sims (LightZero has maintained implementations) |
| Extras | Dynamic Opponent Pool (84%→92.5% over pure self-play), opening diversification | KataGo-style auxiliary targets, initial-state diversification, later regret-guided search control (+77–89 Elo at fixed budget) |
| Product | 1000× cheaper inference → mobile/web | Maximum strength ceiling |

Recommendation: **A first** (cheaper to debug, proven on xiangqi specifically), then B reusing the same env, buffer, and eval.

## Phase 4 — Scale or specialize (decide around month 12)

- **Strength lane:** community scaling — public checkpoints, reproducible configs, distributed self-play client, contributor docs. The only road that ends past full Pikafish.
- **Research lane (unclaimed niches):** Maia-style human-like xiangqi (play like a 1500/1800/2100 human); thinking-tokens xiangqi (search-trace distillation → RLVR — Xiangqi-R1 touched reasoning quality, nobody has done it for strength).
- **Product lane:** mobile/web app on the searchless model.

---

## Eval harness spec (build in Phase 1, keep forever)

- Opponent: Pikafish via UCI `go nodes N`, ladder N ∈ {256, 1k, 4k, 16k, 64k, 256k, 1M}
- Paired games (both colors) from a curated ~100-opening book
- **SPRT** (elo0=0, elo1=20, α=β=0.05) for regression gates; 400-game fixed matches for ladder datapoints; always report Elo ± CI
- Draw adjudication: move cap + engine-checked chase rules
- Cheap per-checkpoint proxies: move-match accuracy split by game phase (opening/middle/end), value-head MSE vs outcomes, policy top-k overlap with Pikafish
- Artifact: the Elo-vs-N chart, updated per model generation

## Risks

| Risk | Mitigation |
|---|---|
| Chase adjudication bugs → RL reward hacking | Rule test suite + engine cross-check before any RL |
| Human PGN data licensing/quality | Prefer engine-annotated data; document sources |
| Phase 2 stalls (hardest engineering) | pyffish intermediate step; C++ fallback lane |
| Scope creep | Every phase exits with a standalone artifact |

## References

- [Mastering Chinese Chess AI (Xiangqi) Without Search — arXiv:2410.04865](https://arxiv.org/abs/2410.04865) (Tencent: recipe + ablations this roadmap borrows)
- [Grandmaster-Level Chess Without Search — arXiv:2402.04494](https://arxiv.org/abs/2402.04494)
- [Chessformer — arXiv:2605.19091](https://arxiv.org/abs/2605.19091) (geometric attention bias, from–to policy head, Maia3)
- [Mastering Chess with a Transformer Model — arXiv:2409.12272](https://arxiv.org/abs/2409.12272)
- [Pgx: Hardware-Accelerated Parallel Game Simulators — arXiv:2303.17503](https://arxiv.org/abs/2303.17503) · [GitHub](https://github.com/sotetsuk/pgx)
- [Superhuman Generals.io via Self-Play RL — arXiv:2606.23348](https://arxiv.org/abs/2606.23348) (3 academics, 4 days, 4 GPUs — what throughput buys)
- [Policy Improvement by Planning with Gumbel (Gumbel MuZero/AlphaZero)](https://openreview.net/forum?id=bERaNdoegnO)
- [Regret-Guided Search Control for AlphaZero — arXiv:2602.20809](https://arxiv.org/abs/2602.20809)
- [LightZero — GitHub](https://github.com/opendilab/LightZero) (maintained AlphaZero/MuZero/Gumbel implementations)
- [Xiangqi-R1 — arXiv:2507.12215](https://arxiv.org/abs/2507.12215)
- [Scaling Scaling Laws with Board Games — arXiv:2104.03113](https://arxiv.org/abs/2104.03113) (train-compute ↔ test-compute exchange rate)
- [KataGo: Accelerating Self-Play Learning in Go — arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
- Pikafish — [GitHub](https://github.com/official-pikafish/Pikafish) · pyffish — [Fairy-Stockfish bindings](https://github.com/fairy-stockfish/Fairy-Stockfish)
