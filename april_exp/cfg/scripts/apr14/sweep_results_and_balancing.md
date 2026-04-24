# CFG apr14 Sweep — Results & Data Balancing Discussion

## 1. What happened: results at tf=0 (pure conjugation)

All runs used 192,231 training examples (3 epochs). Metrics are final-step values, mean ± std over 3 seeds.


| prop | EN_EM       | EN_syn | EN_conf | NL_EM       | NL_syn | NL_conf |
| ---- | ----------- | ------ | ------- | ----------- | ------ | ------- |
| 0.00 | 0.000       | 0.096  | 0.358   | 0.578±0.324 | 0.137  | 0.750   |
| 0.01 | 0.713±0.372 | 0.964  | 0.829   | 0.435±0.213 | 0.042  | 0.751   |
| 0.10 | 0.723±0.354 | 0.999  | 0.899   | 0.465±0.230 | 0.162  | 0.710   |
| 0.25 | 0.882±0.084 | 1.000  | 0.952   | 0.446±0.239 | 0.091  | 0.701   |
| 0.50 | 0.954±0.065 | 1.000  | 0.984   | 0.409±0.326 | 0.202  | 0.663   |
| 0.75 | 0.939±0.054 | 1.000  | 0.953   | 0.197±0.062 | 0.363  | 0.483   |
| 0.90 | 1.000±0.000 | 1.000  | 1.000   | 0.375±0.289 | 0.261  | 0.667   |
| 0.99 | 1.000±0.000 | 1.000  | 1.000   | 0.517±0.189 | 0.196  | 0.749   |
| 1.00 | 0.958±0.060 | 0.990  | 0.979   | 0.000       | 0.956  | 0.505   |


Key observations:

- **EN is easy.** With any non-zero EN data (prop ≥ 0.01), the model reaches near-perfect EN performance.
- **NL is harder and highly variable.** NL exact match ranges from 0.20 to 0.58, with huge variance across seeds (std up to 0.33). This suggests the model is learning NL imperfectly or inconsistently.
- **NL syntax at prop=1.0 ≈ 0.96.** When the model has never seen NL conjugation data, NL outputs still use EN word order (syntax ≈ 1.0 from the EN perspective). This is expected — the model has no NL syntax signal.
- **NL syntax at prop=0.0 ≈ 0.14.** The model is mostly Dutch-ordered but not perfect, and conformity is only 0.75. With no EN training data, performance is generally poor (EN_EM = 0).
- **NL conformity degrades at high prop.** As more training is EN (prop=0.75), NL conformity drops to 0.48, suggesting the model starts generating structurally broken NL outputs.

## 2. What happened: the trans_frac confound

At prop=0.5, varying trans_frac:


| tf   | total examples | total steps (3 ep) | EN_EM | NL_EM | NL_conf |
| ---- | -------------- | ------------------ | ----- | ----- | ------- |
| 0.00 | 192,231        | 9,009              | 0.954 | 0.409 | 0.663   |
| 0.01 | 193,683        | 9,078              | 0.942 | 0.486 | 0.649   |
| 0.10 | 207,816        | 9,741              | 0.988 | 0.834 | 0.969   |
| 0.25 | 236,592        | 11,088             | 0.998 | 0.874 | 0.995   |
| 0.50 | 307,569        | 14,415             | 0.989 | 0.768 | 0.881   |
| 0.75 | 439,386        | 20,595             | 0.973 | 0.819 | 0.951   |
| 0.90 | 591,480        | 27,723             | 0.997 | 0.605 | 0.774   |
| 0.99 | 746,529        | 34,992             | 0.976 | 0.510 | 0.923   |
| 1.00 | 768,924        | 36,042             | 0.000 | 0.000 | 0.495   |


Translation clearly helps NL performance (NL_EM jumps from 0.41 to 0.87 at tf=0.25). But the total training data ranges from 192K to 769K examples — a **4x difference**. This means high-tf models see 4x more gradient updates, making it impossible to attribute improvements to translation signal vs. more training.

## 3. How data is currently generated

The pool of sentence pairs is fixed: **192,231 train pairs** (from 40K unique trees per structure × 6 structures, 80% train split). Each pair has `en_present`, `en_perfect`, `nl_present`, `nl_perfect`.

Given `trans_frac` and `prop`:

1. Split the 192K pairs into two pools: `n_trans` pairs for translation, the rest for conjugation.
2. **Translation pairs** → 4 examples each (tense_separate: present→present + perfect→perfect, in both directions en2nl and nl2en).
3. **Conjugation pairs** → 1 example each (either EN or NL, selected by `prop`).

The formula for `n_trans` is chosen so that the *fraction of examples* that are translation equals `trans_frac`:

```
n_trans = round(trans_frac * P / (e - trans_frac * (e - 1)))
```

where `e = 4` (examples per translation pair). This means:

- At tf=0.5: half the pairs go to translation (producing 4x), half to conjugation (1x) → total = 0.5P + 0.5P×4 = 2.5P
- At tf=1.0: all pairs go to translation → total = 4P

The 4:1 fan-out is the root cause of the imbalance.

## 4. Options for balancing

### Option A: Cap total examples at N_base (your proposal)

Fix the total number of training examples at `N_base = 192,231` (the tf=0 count) regardless of trans_frac.

How it works: After generating all conjugation + translation examples, randomly subsample down to N_base. This preserves the *ratio* of translation to conjugation but caps total data.

Example at tf=0.5 (prop=0.5):

- Currently: 153,785 conj + 153,784 trans = 307,569
- Capped: subsample to 192,231 → ~96K conj + ~96K trans

**Pros:** Simple. Total data and training steps are identical across all runs. Clean comparison.
**Cons:** At high tf, you're throwing away a lot of translation data (tf=1.0 would drop from 769K to 192K). More importantly, at high tf you now have *fewer* conjugation examples than at tf=0 (because translation ate into the pair pool). At tf=0.5 capped, you'd have ~96K conjugation examples vs. 192K at tf=0 — so the model sees less of the primary task.

### Option B: Fix conjugation count, add translation on top with a budget

Keep conjugation examples fixed at ~192K (i.e., use ALL pairs for conjugation). Then *separately sample* translation examples from the same pairs up to a budget.

How it works:

1. All 192K pairs → 192K conjugation examples (split by prop as before).
2. Independently, sample `k` pairs from the same pool and generate translation examples from them. Cap `k` so that total = N_base, i.e., `k = 0` (no translation budget left since conjugation already uses all N_base) — this doesn't work because it means no room for translation.

Alternative: fix total at something like 1.5 × N_base to allow some translation headroom, but this is arbitrary.

**Verdict:** Doesn't cleanly work if we want identical totals.

### Option C: Fix total examples, adjust the split

Fix total at N_base. Compute how many pairs to allocate to translation such that `conj_examples + trans_examples = N_base`:

```
Let t = number of pairs for translation
    c = P - t (pairs for conjugation → c examples)
    trans_examples = 4t
    total = c + 4t = (P - t) + 4t = P + 3t = N_base
    → 3t = N_base - P = 0  (since N_base = P)
    → t = 0
```

This shows that if N_base = P and each translation pair makes 4 examples, you can't fit *any* translation without exceeding N_base. You'd need N_base > P.

To make this work, set `N_target = P` (192K) and *subsample* within each pool:

- Allocate `t` pairs to translation → `4t` translation examples
- Remaining `P - t` pairs available for conjugation
- Subsample conjugation down to `N_target - 4t` examples
- Constraint: `N_target - 4t ≥ 0` → `t ≤ N_target / 4 = 48K` (max tf ≈ 0.25 of pairs)

For trans_frac as a fraction of *examples*:

- tf=0.25 at 192K total → 48K trans examples + 144K conj examples → need 12K trans pairs → 180K conj pairs available (only use 144K)
- tf=0.50 at 192K total → 96K trans examples + 96K conj examples → need 24K trans pairs → 168K conj pairs available (only use 96K)
- tf=0.75 at 192K total → 144K trans examples + 48K conj examples → need 36K trans pairs → 156K conj pairs available (only use 48K)
- tf=0.90 at 192K total → ~173K trans examples + ~19K conj → need ~43K trans pairs → 149K conj pairs available (only use 19K)
- tf=1.0 → 192K trans examples, 0 conj → need 48K trans pairs

This works cleanly: total is always exactly 192,231.

**Pros:** Clean. Identical data budget. trans_frac cleanly controls the mix.
**Cons:** At high tf, conjugation examples drop dramatically (48K at tf=0.75 vs. 192K at tf=0). This is inherent to the fixed-budget design — you're trading conjugation data for translation data. Whether this is a confound depends on the question: if the question is "does translation help conjugation *given a fixed training budget*", this is the right design. You're asking: "if I can only afford N tokens of training, should I spend some on translation?"

### Option D: Fix number of *conjugation* examples + control translation separately

Keep conjugation fixed at `C` examples regardless of tf. Then add translation on top:

- tf=0: C conj + 0 trans = C total
- tf=0.5: C conj + C trans = 2C total  
- tf=1.0 doesn't make sense (no conjugation)

This isolates the question: "does *adding* translation help, holding conjugation constant?" But total data varies (C to 2C).

### Option E: Fix total *gradient updates* (steps), not examples

Instead of fixing example count, fix the number of training steps. With more data, train fewer epochs (or use `max_steps` instead of `num_train_epochs`).

For example, fix at 9,009 steps (the tf=0 count). At tf=1.0 with 769K examples, that's only 9009/12014 ≈ 0.75 epochs instead of 3.

**Pros:** Every model sees the same number of weight updates.
**Cons:** High-tf models see each example fewer times (less memorization), low-tf models see each example more. Whether this is "fair" depends on your theory of what matters.

## 5. Recommendation

**Option C** (fix total examples at 192K, subsample within each pool) is the cleanest design for the question: *"Given a fixed training budget, does allocating some of it to translation improve conjugation performance?"*

Implementation: in `build_training_data`, after generating all conjugation and translation rows, if `len(all_rows) > N_target`, subsample `all_rows` down to `N_target` while preserving the trans_frac ratio. Or more precisely: compute how many conjugation examples to keep as `N_target - n_trans_examples`, and subsample the conjugation pool accordingly.

This requires a one-line change in `dataset_manager.py`: after building `conj_selected` and `trans_rows`, cap `conj_selected` to `N_target - len(trans_rows)`.