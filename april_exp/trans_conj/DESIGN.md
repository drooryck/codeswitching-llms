# april_exp Design Document

Two changes to the training pipeline from `march_exp`.

---

## Change 1: Task token replaces `<sep>`

### Motivation

David's suggestion: the token that separates input from target should also
carry the task identity.  This is more efficient (one fewer special token in
every sequence) and may give the model a cleaner signal — it sees the full
input *before* learning what task to perform.

### Old format (march_exp)

```
<sos> <conjugate> input_sentence <sep> target_sentence <eos>
```

- Task token at position 1 (after `<sos>`)
- `<sep>` at the input/target boundary
- Loss masked over `[<sos>, <task>, input..., <sep>]`

### New format (april_exp)

```
<sos> input_sentence <conjugate> target_sentence <eos>
```

- No leading task token
- The task token (`<conjugate>` / `<translate>`) sits WHERE `<sep>` was
- It simultaneously identifies the task AND marks the boundary
- Loss masked over `[<sos>, input..., <task_token>]`

### Examples

Conjugation:
```
<sos> le chat mange le chien <conjugate> le chat a mangé le chien <eos>
```

Translation (tense_separate):
```
<sos> le chat mange le chien <translate> de kat eet de hond <eos>
```

Translation (full_sequence):
```
<sos> le chat mange le chien <pair> le chat a mangé le chien <translate> de kat eet de hond <pair> de kat heeft de hond gegeten <eos>
```

### Inference prompt

```
<sos> input_sentence <task_token>
```

The model generates everything after the task token (the target sentence +
`<eos>`).

### Implementation

- **`dataset_manager.py` `encode()`**: builds `[bos] + input_ids + [task_token_id] + target_ids + [eos]`, masks labels up to and including the task token position.
- **`experiment.py` `_run_inference()`**: prompt is `<sos> {input} {task_token}`, prediction extracted by splitting on the task token.
- **`<sep>` token**: kept in the tokenizer vocabulary for backward compatibility with `model_config.py`'s `tokenizer_config`, but never appears in training data or inference.

---

## Change 2: Proportion tracking with exact token counts

### What `prop` controls

`prop` = fraction of French examples within the conjugation task (same
knob as march_exp).  Translation is always balanced (50/50 fr2nl/nl2fr).

### What we report (exact)

After building the dataset, every token in every input and target string
is matched against the FR and NL word sets from the lexicon.  We then
report **exact counts**:

| Stat | Meaning |
|------|---------|
| `conj_fr_token_frac` | Fraction of FR tokens within conjugation data |
| `trans_fr_token_frac` | Fraction of FR tokens within translation data |
| `overall_fr_token_frac` | Fraction of FR tokens across all data |

These are logged to both the experiment log and wandb for every run.

### Why not derive `prop` from a target overall fraction?

Translation data is always roughly 50/50 FR/NL tokens (each example has
one sentence per language).  So the overall FR token fraction is:

```
overall ≈ (prop × conj_tokens + 0.5 × trans_tokens) / total_tokens
```

The overall proportion is simply the conjugation proportion *dampened*
toward 0.5 by the translation data.  Rather than trying to "undo" this
dampening (which requires clamping, approximations, and edge-case
handling when trans_frac is large), we keep `prop` as the direct knob
for conjugation and report the resulting overall fraction exactly.

---

## Design question: Should we control the translation direction balance?

### The question

Currently translation is always balanced: equal numbers of `fr2nl` and
`nl2fr` examples.  Should we allow breaking this symmetry to gain finer
control over the French token proportion?

### Analysis

Consider what each translation direction contributes:

| Direction | Input (masked) | Target (loss) | All tokens |
|-----------|----------------|---------------|------------|
| `fr2nl`   | French         | Dutch         | ~50% FR    |
| `nl2fr`   | Dutch          | French        | ~50% FR    |

**Key insight:** From an all-tokens perspective, both directions contribute
~50% French tokens.  Breaking the symmetry does NOT meaningfully change
the total French token proportion.

However, there is a subtlety with **loss masking**.  The model only
receives gradient signal from the unmasked target tokens:

| Direction | Loss signal language |
|-----------|---------------------|
| `fr2nl`   | Dutch (NL target)   |
| `nl2fr`   | French (FR target)  |

So the direction balance *does* affect what language the model is
**trained to produce**, even though it doesn't change the total French
tokens seen.

### Two valid definitions of "French proportion"

1. **All-tokens**: proportion of French tokens the model sees in its
   forward pass (both masked input and unmasked target).  This is what
   the current `prop` controls.

2. **Loss-tokens**: proportion of French tokens in the unmasked (target)
   portion only — i.e., the language the model is trained to generate.

These diverge for translation examples.  For conjugation, they are
identical (input and target are the same language).

### Recommendation

**Keep translation balanced (50/50)** for these reasons:

1. **Simplicity**: One parameter (`prop`) controls one thing.  Adding a
   translation balance parameter creates a 2D design space that is harder
   to sweep and interpret.

2. **Semantic coherence**: Breaking translation symmetry means the model
   gets more practice translating in one direction than the other.  This
   confounds the language-proportion manipulation with a task-balance
   manipulation.

3. **Minimal gain**: Since both directions contribute ~50% French tokens,
   breaking symmetry barely moves the overall proportion.  The
   conjugation balance is a much more effective lever.

4. **Loss-signal consideration**: If the goal is to study what the model
   *produces* (codeswitching in generation), then the loss-token
   perspective matters.  But conjugation — where the model produces
   monolingual output — is the primary evaluation task.  The translation
   task exists to provide cross-lingual signal, not as an end in itself.

If future experiments need to study the effect of translation direction
imbalance specifically (e.g., "does more fr2nl exposure cause more NL
generation?"), that can be added as a separate parameter without
conflating it with the overall proportion control.

---

## File changes summary

| File | Changes |
|------|---------|
| `translation.py` | Docstring updated; no functional change (format functions don't touch special tokens) |
| `dataset_manager.py` | `encode()` uses new boundary-task-token format; `build_training_data()` derives conjugation balance from target overall proportion via `_derive_conj_prop()` |
| `experiment.py` | `_run_inference()` prompt changed to `<sos> input <task_token>`; prediction extraction splits on task token instead of `<sep>` |
| `run_single.py` | Updated help text for `--prop` |
| `model_config.py` | Copied verbatim |
| `metrics.py` | Copied verbatim |
