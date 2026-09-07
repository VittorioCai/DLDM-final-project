# Invariants the test suite must enforce

Every check below runs as an automated `pytest` test rather than a manual
spot check, because the counterexample design fails silently: a dataset with
the wrong conflict count still trains and still reports plausible accuracy.

1. `test_total_size_constant` — every budget trains on exactly 50,000 images.
2. `test_digit_balance` — every digit contributes exactly 5,000 images.
3. `test_exact_counterexample_budget` — the realised conflict count equals N
   exactly, and each class contributes N/10.
4. `test_color_marginal_constant` — the global count of each of the ten hues
   is identical across budgets.
5. `test_nested_counterexamples` — within a seed, the conflict indices of a
   smaller budget are a subset of those of a larger one.
6. `test_seed_reproducibility` — the same seed reproduces results item by
   item; different seeds differ at least in index permutation.
7. `test_counterfactual_shape_invariance` — recoloring leaves the grayscale
   mask, the label and the sample id unchanged.
8. `test_all_hues_rendered_once` — each test shape yields exactly ten colored
   versions.
9. `test_resnet_output_shape` — input `[B,3,28,28]` gives output `[B,10]`.
10. `test_metric_known_examples` — flip rate, counterfactual consistency and
    worst-group accuracy are verified against hand-constructed predictions
    with known answers.

Beyond the automated suite, save one 10x10 image grid before training: ten
digits by ten hues. Confirm by eye that digit shapes, labels, the hue mapping
and background handling are all correct. An automated test cannot catch a
palette that is subtly wrong to a human observer.
