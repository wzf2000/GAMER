# Development Documentation

Use this directory for planning, design notes, decisions, and development records that should live on the `dev` branch.

## Layout

- `roadmap.md` / `roadmap.zh.md`: long-term project plan, milestones, priorities, and active work.
- `design/`: design documents for specific features, experiments, or refactors.
- `decisions/`: decision records for important technical or workflow choices.
- `notes/`: temporary development notes, debugging records, and experiment observations.

## Current Index

- `design/2026-06-01-paper-code-map.md` / `.zh.md`: maps the arXiv GAMER paper concepts to the current code implementation, baselines, and train/test pipeline.
- `design/2026-06-01-temporal-hierarchical-behavior-modeling.md` / `.zh.md`: proposes the next GAMER method direction around temporal-hierarchical behavior modeling and cross-level module revisions.
- `design/2026-06-09-th-results-and-next-design.md` / `.zh.md`: summarizes TH variant results and recommends follow-up model design directions.
- `decisions/2026-06-01-temporal-hierarchical-method-direction.md` / `.zh.md`: records the accepted direction for input representation, attention architecture, and layer allocation in the next GAMER revision.

## File Names

Use Markdown files named with this pattern:

```text
YYYY-MM-DD-short-topic.md
```

Example:

```text
2026-06-01-branch-and-rules-policy.md
2026-06-01-branch-and-rules-policy.zh.md
```

## Bilingual Rule

Keep `docs/dev/` Markdown documents in English and Chinese pairs. English documents use the normal `.md` suffix; Chinese counterparts use `.zh.md`. If a document is first written in Chinese, add the English counterpart without `.zh`. Update both versions when adding, renaming, or materially changing development docs.
