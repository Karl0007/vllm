# #53142 restore-fidelity arms for #55507

Adapted fixture and the three arms behind our [comment on #53142](https://github.com/vllm-project/vllm/issues/53142#issuecomment-5779611799),
pushed so the adaptation and the fix can be reviewed independently.

Base of this branch is `MaCoredroid:vllm:p8-restore-fidelity` (the restore-fidelity
fixture plus #53798's implementation). Everything added here lives under
`oracle-53142/`; no source file of the base is modified.

## The three arms

| arm | seeding divisor while `_mamba_spec` is bound / unbound | `mamba_hybrid.py` in the tree |
|---|---|---|
| A | `MambaSpec.block_size` / n.a. (resolved in `set_kv_cache_config`) | base (unchanged) |
| B | `_mamba_spec.block_size` / `cache_config.block_size` | `mamba_hybrid_armB_55507_asis.py` |
| C | `_mamba_spec.block_size` / `cache_config.mamba_block_size or cache_config.block_size` | `mamba_hybrid_armC_55507_fixed.py` |

B is #55507 as written (the state before `adc7d30`); C is the same code with the
fallback fix. Both B and C carry #55507's *lazy* binding, i.e. the spec is
resolved on first use instead of in `set_kv_cache_config`.

## Adaptation: what "preserves their production binding" means for #55507

Three changes to the fixture, none of which touch what is under test:

1. `set_kv_cache_config` becomes a no-op that stores the config -- #55507 resolves
   `MambaSpec.block_size` lazily, and base `main` has no `set_kv_cache_config` at
   all, so the fixture cannot run against main unmodified.
2. The fixture seeds the attributes `__init__` would set (`_mamba_spec = None`,
   `_mamba_group_ids`, `_mamba_ctx`, `_mamba_state_copy_funcs`), because it
   bypasses `__init__` via `object.__new__`.
3. `cache_config` carries `mamba_block_size`, which production has and the
   fixture's `SimpleNamespace` did not.

## Results (4x CMP 170HX, sm_80, one card, CUDA)

| arm | result |
|---|---|
| A | **10 passed** |
| B | **2 failed**: `test_add_request_seeds_state_idx_in_mamba_blocks` (global-unit column, 6709 vs the expected 121) and `test_align_resume_restores_committed_state_bitwise[unequal_geometry]` (`expected block 6 (column 2) restored into block 5 (column 3) ... blocks differing from that image: [5]`); `equal_geometry_control` and both negative controls passed |
| C | **10 passed** |

The full-file counts include the fixture's unrelated tests, which pass in every arm.

## Running it

```bash
./oracle-53142/run.sh A    # tree as-is            -> 10 passed
./oracle-53142/run.sh B    # #55507 as written     -> 2 failed, 8 passed
./oracle-53142/run.sh C    # #55507 + fallback fix -> 10 passed
```

`run.sh` copies the compiled extensions out of `IMAGE` into the tree (a
sparse/partial checkout has none) and runs pytest with `--noconftest`: the test
file is self-contained and the image ships no `tblib`. Set `CUDA_VISIBLE_DEVICES`
to pick the card; the defaults assume an image with vLLM's dependencies.

## Scope

The fixture checks the seeding unit and the restored state bytes. It does not
establish equivalence of the two variants' initialization paths -- that is
exactly where B and C differ from A, and the difference is visible only through
the arms above. We have not run `p9-mamba-aligned-state-indices` here.

AI-authored (agent-written, verified on our hardware; the account owner is a
user, not a vLLM developer).
