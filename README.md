# Predicting expected substitutions
The Python module containing functions needed for modelling expected substitutions and validating/visualising results and model statistics against the gold standard method - BEAST. Accompanying the "Mind your Ps: Using Probability in the Interpretation of Molecular Epidemiology Data" PHE manuscript and QMUL thesis. This project has been carried out in the context of my work at Public Health England and my MSc in Bioinformatics at Queen Mary University of London.

For up-to-date project, go to [PHE GitHub repo](https://github.com/phe-bioinformatics/mind-your-ps-2021-manuscript-code.git/).

A model describing expected substitutions in probabilistic terms: stochastic
variation of a substitution rate.
A BEAST tree is the gold standard analysis for taking into account sequence and
time of sampling in the clustering of samples. However BEAST is time-, computation-,
and knowledge-intensive. Alternatively, a rough simplified way of excluding
transmission chains (without using BEAST) for routine use or for those who sequence
a smaller fraction of the circulating strains and/or lacking in computing
resources could be based on a simpler probabilistic approach.

1. Measles virus N-450 and MF-NCR sequences (B3, D4, D8 genotypes)
2. Sequences curated in BioNumerics v6.1 (BN) IDU measles WHO database
3. Seqs exported using the BN plugin for exporting fastas (BN keys as seq IDs)
4. Sample info exported from BN using data export plugin
5. IQ-TREE phylogenetic analyses
6. BEAST phylogenetic analyses
7. Pairwise comparison
8. Expected substitutions per week based on a Poisson distribution
9. BEAST estimates compared to model predictions to evaluate fit

## Model
Given a substitution rate (# substitutions/time) applicable to a specific genomic
region, then expected substitutions after time t will be rate * t. Upper and lower
intervals of confidence to X% can be determined using an exponential distribution
of a poisson event with rate=substitution rate.

In practical terms, the expected number of substitutions for a region of the genome
can be plotted in relation to time. Upper and lower expected #substitutions can
be plotted at a given confidence interval. Then, a pair of samples for which the
sequences and onset/collection dates are known can be plotted over the previous
graph where the x-coordinate will be proportional the time difference between
samples and the y-coordinate the #differences between the samples' sequences.
If the pair representation falls out of the expected number of differences for
the time between them for a CI of X% then one can exclude that the samples are
within the chain of transmission of interest with X% confidence.

Given the absence of a tree, this Hamming distance will be an approximation of
the real divergence between the samples. However, this will over-estimate
divergence:
```
  real tree           assumed tree
   |---- A              A
---|                 ---|               time since MRCA rt > at (substitutions are known)
   |---B                |-------B
```
Hence, it is important to take in consideration the time to the most recent
common ancestor (MRCA).
e.g.,
* subst rate 1 site/week, at 4 weeks 4 differences +/- 1 > 3-5 substitutions would be
  expected if the samples shared a common ancestor at time 0
* samples A and B 8 differences, 4 weeks apart
* if we simply count subs vs weeks we are saying  the 8 diffs were accumulated
  in 4 weeks > we would exclude endemic transmission given  that 8>5 (upper limit
  of expected subst).
  > *case A:*  B direct descendant of A
  ```
  A -------- B     t=4 d=8
  ```
* However, commonly A and B derive from a MRCA.
  > *case B:* MRCA before A or B
    ```
        |--- A
      --|            t>4 d=8 (more time to accumulate the 8 substitutions)
        |----- B
     ```
Our null hypothesis is:
> H0: A and B share a common ancestor MRCA within the time frame within which the
MRCA was detected.

So we want to err on the side of caution and not exclude that A and B descend
from the MRCA when we do not have sufficient information rather than assume
that B descends directly of A.

**How to correct the values?**
*A and B are contemporaneous samples:*
```
        |----- A
      --|
        |----- B
        t0    tA = tB
```
The maximum time between A or B and the MRCA for them to be endemic would be
t = tA - t0 = tB - t0 weeks .
Hence both A *and* B had t weeks to accumulate the observed number of
substitutions. We would conservatively assume that S substitutions could have
occurred over the cumulative evolution time of 2 x t weeks.

*A and B are x weeks apart*
```
        |-------- A
      --|
        |---------------- B
        t0       tA      tB
```
In this case, A could have had t = tA - t0 weeks since the MRCA to accumulate
substitutions, while B could have had t' = tB - t0. We would conservatively
assume that S substitutions could have occurred over the cumulative evolution
time of t + t' weeks.

*A and B are t' weeks apart*
```
        | A
      --|
        |------------------------------- B
        t0 tA                           tB
```
Now, following the previous thought process, B could have had t' weeks to
accumulate S substitutions, while A would have had none. However, if we have A
and a MRCA sample collected at the same time, we must assume we missed at least
one common ancestor of those two samples. Hence the time to a MRCA cannot be t',
but needs to account for the infectious period between the CA and A/MRCA. The
time to account for is then t' + 2*infectious periods. We would conservatively
assume that S substitutions could have occurred over the cumulative evolution
time of t' + 2*infectious periods weeks.

**Conclusion:** *To conservatively correct for an unknown time to a MRCA,
we must add up the time difference between each sample and the MRCA. When A was
contemporary to the MRCA, a missed CA must be assumed*

## Validation
To validate the probabilistic model, samples from a BEAST tree can be
plotted onto the time/differences plot and colour-coded them by BEAST distance
or groupings.

### Definitions
H0: Two sequences are related within the time frame given

#### Model outcomes
* False positive (FP) = type I error = true H0 rejected,
     * seqs related, but predicted as unrelated
     * likely (L) in BEAST, but above (A) expected #subs in model - LA
     * the error to avoid here
* False negative (FN) = type II error = false H0 non-rejected
     * seqs unrelated, but predicted as related
     * unlikely (U) in BEAST, but below (B) or within the expected range (E)
       of #subs in model - UB + UE
* True positive (TP) = false H0 rejected
     * seqs not related and predicted as not related
     * unlikely (U) in BEAST and above (A) expected #subs in model - UA
     * including possibles: UA + possible (P) in BEAST and above (A) expected
       subs in model - UA + PA
* True negative (TN) = true H0 accepted
     * seqs related and predicted as related
     * likely (L) in BEAST and below (B) or within the expected range (E)
       of #subs in model - LB + LE
     * including possibles: LB + LE + possible (P) in BEAST and below (B) or
       within the expected range (E) of #subs in model - LB + LE + PB + PE

#### Rates
* True positive rate (TPR)
    * % positive pairs classified as positive (*sensitivity*)
    * UA / U
    * including possibles: (UA + PA) / (U + P)
* True negative rate (TNR)
    * % negative pairs classified as negative (*specificity*)
    * (LB + LE) / L
    * including possibles: (LB + LE + PB + PE) / (L + P)
* False positive rate (FPR)
    * % negative pairs classified as positive
    * LA / L
* False negative rate (FNR)
    * % positive pairs classified as negative
    * (UB + UE) / U
* Accuracy (ACC)
    * % pairs correctly classified
    * (TP + TN) / total = (UA + LB + LE) / (L + U)
    * including negatives:
    (TP + TN) / total = (UA + LB + LE + PB + PE + PA) / (L + U + P)
* False discovery rate (FDR)
    * % of pairs classified as positive that are in fact negative
    * LA / A
* False omission rate (FOR)
    * % of pairs classified as negative that are in fact positive
    * (UB + UE) / (B + E)
* Negative predictive value (NPV)
    * % of pairs classified as negative that are really negative
    * (LB + LE) / (B + E)
* Positive predictive value (PPV)
    * % of pairs classified as positive that are really positive
    * UA / A
