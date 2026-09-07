# Reading the final results

## The short answer

The number of counterexamples needed to break the color shortcut is not a
constant. It depends jointly on the training method and the model
architecture. Under the prespecified N80 criterion (bias-conflict accuracy
>= 80% and flip rate <= 10%), the five-seed medians of the six combinations
fall between **250 and 1,000 counterexamples**.

## N80, the primary result

| Model | Method | Median N80 | Seed range | Five seeds |
|---|---|---:|---:|---|
| LeNet | ERM | 1000 | 1000-2500 | 1000, 2500, 1000, 1000, 2500 |
| LeNet | GroupDRO | 500 | 250-500 | 500, 500, 250, 250, 500 |
| LeNet | DFR | 250 | 250-250 | 250, 250, 250, 250, 250 |
| ResNet-18 | ERM | 1000 | 500-2500 | 1000, 1000, 500, 1000, 2500 |
| ResNet-18 | GroupDRO | 500 | 250-500 | 500, 500, 250, 250, 500 |
| ResNet-18 | DFR | 1000 | 500-1000 | 1000, 1000, 1000, 500, 1000 |

## Three findings that matter

### 1. With no counterexamples the models are right for the wrong reason

Under ERM at N=0:

- LeNet reaches 99.99% aligned accuracy, but only 7.11% bias-conflict
  accuracy, with a 92.89% flip rate.
- ResNet-18 reaches 98.96% aligned accuracy, but only 5.17% bias-conflict
  accuracy, with a 94.68% flip rate.

An ordinary aligned test sees almost nothing wrong. Only the counterfactual
test, which holds the shape fixed and changes the hue, exposes the shortcut.

### 2. GroupDRO is the most stable across architectures

GroupDRO brings the median N80 down to 500 on both models, with the same
250-500 seed range. It is not the best method at every budget, but it is the
least sensitive to the change of architecture.

### 3. DFR shows a clear architecture interaction

DFR is most effective on LeNet, where all five seeds pass at N=250. On
ResNet-18 its median N80 rises to 1,000, while GroupDRO on the same backbone
needs only 500.

So the result cannot be written as "DFR is always best". The accurate
statement is: **DFR's sample efficiency depends strongly on the
representation it is refitting, whereas GroupDRO is more stable across the
two architectures tested.**

## Paired method differences at N=500

Mean differences relative to ERM at the same model, budget and seed:

| Model | Method | d conflict accuracy | d flip rate | d aligned accuracy |
|---|---|---:|---:|---:|
| LeNet | GroupDRO | +8.62 pp | -8.80 pp | -0.30 pp |
| LeNet | DFR | +10.19 pp | -10.58 pp | -0.52 pp |
| ResNet-18 | GroupDRO | +7.28 pp | -7.25 pp | +0.10 pp |
| ResNet-18 | DFR | +2.24 pp | -2.19 pp | +0.12 pp |

`pp` is percentage points. Five seeds is few, so these numbers describe
effect sizes and are not dressed up as significance tests.

## Sensitivity

- Lowering the accuracy threshold from 80% to 70% leaves all six median
  thresholds unchanged, because most conditions are bound first by the
  flip rate <= 10% requirement.
- Raising it to 90% moves only LeNet-ERM, from 1,000 to 2,500. Every other
  median is unchanged.
- GroupDRO's stability and DFR's architecture dependence are therefore not
  artifacts of the single 80% cutoff.

## How to state this in writing

Defensible:

> In controlled Colored MNIST experiments, the number of counterexamples
> required to break the color shortcut depended jointly on the learning
> method and model architecture. GroupDRO reached the N80 criterion with a
> median of 500 counterexamples for both LeNet and ResNet-18, whereas DFR
> required 250 for LeNet but 1,000 for ResNet-18.

Not supported by this data:

- "250 counterexamples always break the shortcut."
- "Larger models are always harder to debias."
- "DFR always outperforms GroupDRO."
- "These results show that all real vision models behave this way."

## Scope

These conclusions hold for controlled Colored MNIST, this particular
ten-color palette, two CNN architectures, three methods and the current
budget grid. They do not extrapolate to natural images, to multiple
interacting shortcuts, or to settings where group labels are unknown.
