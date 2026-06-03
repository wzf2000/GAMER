# BatchEncoding Attribute-vs-Dict Gotcha

## Context

While doing the P4 refactor of `test_MB_decoder` (Tier 2 follow-up to the test
task consolidation), the goal was to migrate the Qwen3Multi inline `generate`
call to the shared `build_generation_kwargs` helper. Runtime verification on a
1-epoch Qwen3Multi Toy MB checkpoint hit a tensor-shape mismatch inside
`Qwen3MultiModel._update_session_multi_cross_mask`:

```
RuntimeError: The size of tensor a (21) must match the size of tensor b (20)
              at non-singleton dimension 3
```

The inline call worked; the helper-built call did not — despite producing what
looked like identical kwargs.

## Root Cause

`transformers.BatchEncoding` overrides `__getattr__` to fall through to the
underlying `data` dict, but a plain attribute *set* sticks on the Python object
without writing back into the dict.

`test_MB_decoder.test_single_type` does exactly this in the
`TARGET_BEHAVIOR`/`BEHAVIOR_SPECIFIC` branch:

```python
action = [[dataset.behavior_level[u]] for u in behaviors]
inputs.actions = torch.cat([inputs.actions, torch.tensor(action, device=...)], dim=1)
```

After this line:

| access method | returns |
|---|---|
| `inputs.actions` | new extended tensor (length 21) |
| `inputs["actions"]` | original collator tensor (length 20) |
| `inputs.get("actions")` | original collator tensor (length 20) |
| `getattr(inputs, "actions", None)` | new extended tensor (length 21) |

The original inline `generate(..., actions=inputs.actions, ...)` saw the new
length-21 tensor. The first version of the tolerant helper used
`inputs.get("actions")` and silently picked up the stale length-20 tensor,
which then mismatched the input_ids length inside the model's mask builder.

## Fix

Use `getattr(inputs, name, None)` everywhere inside helpers that read optional
fields from a `BatchEncoding` originating from a test task that may
locally reassign attributes:

```python
if backbone_uses_actions(backbone):
    actions = getattr(inputs, "actions", None)
    if actions is not None:
        gen_kwargs["actions"] = actions
```

`getattr` defers to BatchEncoding's `__getattr__`, which respects both the
explicit attribute assignment and the dict fallback, and returns the
default when the field is genuinely missing (e.g. MB datasets have no
`session_ids`).

## Verification

Same Qwen3Multi Toy MB checkpoint and the same script invocation produce
byte-identical `results-original.json` plus per-uid `user_level_metrics_*.json`
before and after the fix.

## Rules of Thumb

- Don't use `inputs.get(key)` / `inputs[key]` inside helpers that may be called
  after a test task mutates `inputs.<field>`. The dict can lag behind the
  attribute.
- If a helper sets a field back into `inputs`, do it consistently with the
  surrounding code. The current style is attribute assignment
  (`inputs.actions = ...`), which means the dict is intentionally not the
  source of truth for mutated fields.
- For collated-but-optional fields (e.g. `session_ids`, which the MB collator
  never produces), `getattr(inputs, name, None)` keeps the helper portable
  across MB / SMB / sequential datasets.

## See Also

- Commit `52e8bfb` — applies the fix and removes the Qwen3Multi inline branch.
- `SeqRec/tasks/evaluation/helpers.py::build_generation_kwargs` — the helper
  the fix lives in.
- `SeqRec/tasks/evaluation/test_MB_decoder.py::TestMBDecoder.test_single_type`
  — the call site that mutates `inputs.actions`.
