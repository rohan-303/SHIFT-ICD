# STEP 11-R2P2 — FORM-WEIGHTING PROTOCOL AMENDMENT

## Status

`STEP11_R2_EXECUTION_PROTOCOL_COMPLETE_FROZEN`

The amendment occurred before feature-cache materialization, training, checkpoint creation, FUSION_VAL scientific scoring, official DEV access, and TEST access.

## R2-X001 closure

The previous blocker identified that `TRAIN_DERIVED_FORM_WEIGHTED` lacked an executable weighting formula, scaling rule, loss scope, and zero-count behavior. Those fields are now frozen in `form_weighting_contract_v1.json`.

## Frozen formula

For class `c`, `w_c = N / (C * n_c)`, using only FUSION_TRAIN source counts. `N=8666`, `C=6`, and the class order is the authoritative six-form order. Computation and manifest storage use float64. The weighted-average property `sum_c(n_c*w_c)/N = 1` was verified; no additional normalization is applied.

## Counts and weights

{
  "counts": {
    "ALTERNATIVE": 1613,
    "COMBINATION": 190,
    "COMBINATION_WITH_ALTERNATIVES": 207,
    "NO_MAP": 251,
    "SINGLE_APPROXIMATE": 4310,
    "SINGLE_EXACT": 2095
  },
  "weights_float64": {
    "ALTERNATIVE": 0.8954329406902253,
    "COMBINATION": 7.601754385964912,
    "COMBINATION_WITH_ALTERNATIVES": 6.977455716586151,
    "NO_MAP": 5.754316069057105,
    "SINGLE_APPROXIMATE": 0.3351121423047177,
    "SINGLE_EXACT": 0.6894192521877486
  }
}

The weighting applies only to `L_form`. Structural count, activity, slot, assignment, and flat-set losses are unchanged. The unweighted variant uses six unit weights. A zero-count required class raises `STEP11_FORM_WEIGHT_UNDEFINED_ZERO_CLASS`.

## Versioned contracts

- `form_weighting_contract_v1.json`: `eac3a1bd0b0848b9e7635c68dd0153a6b33b5c0dfab8e0e2f74264482a5ed6a4`
- `loss_contract_v3.json`: `f1a246b581199e45cf06f2cf9b39bb084977ca7cd1b86b23921b2bbf1e77e959`
- `search_space_v4.json`: `69bec96cb49ffac41e3352820097e47f45878c916b00080c39fbcc84fea54456`
- `r2_configuration_grid_v3.json`: `de2e98fd71df1ae2026e43a2a3b973cdbcdbe2da463fed3628b8487f4de445c5`
- `r2_search_protocol_v3.json`: `c21491883eed378f9568823ac9a22554a84ab5ae56f85fc6df2f26c3bdd340d5`
- amendment manifest: `4752b75ab321717e4aa769b3d49b5ef6dd020b2badbfe11112169b600d6c1a76`
- complete preflight: `88ea6a266acb6f828d19e39561af8192a44bb924ec5d702c39b91f53302ee24c`

Metric, selection, and interpretation contracts remained unchanged: `{'metric_contract_v2_sha256': '19fb3122266a3c6f11cec7e58d1bb0f5860faf2dbba113d7c3d099037de740a6', 'selection_rule_sha256': 'f17bd93e0a864070ee13b015c3dc72770ab5c3444f42ebbcecf2ca7eb98cdc8a', 'interpretation_contract_sha256': 'ecad6ee74d4bd48e3efca7f45597de729dae0fb8079c0dd8a62087c9f9efb5ef'}`.

## Scientific grid preserved

The grid remains four configurations: two learning rates (`0.0001`, `0.0003`) × two imbalance variants. The future execution remains three epochs per configuration, for 12 expected scientific epoch rows. No R2 scientific result exists.

## Validation

TRAIN source count: 8,666; source SHA: `150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c`. FUSION_VAL remains 1,531 sources with source SHA `225df0f8343bc5358e1980688018916df4c75675595a77f37ca65ceea17bbcb9` and zero overlap. DEV and TEST remain quarantined. Full R2 feature caches: 0; configurations trained: 0; epoch rows: 0; checkpoints: 0; FUSION_VAL scientific evaluations: 0; official DEV evaluations: 0; TEST scoring: 0.

The amended protocol authorizes a future R2 execution preflight, not R2 training in this milestone.
