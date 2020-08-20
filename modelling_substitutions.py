# code by Ana.Penedos@phe.gov.uk 
# last updated July 2020

# standard python libraries
from itertools import combinations
import math
import os
import re
# 3rd party python libraries
from Bio import SeqIO
import dendropy as ddp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson


# permissive matching, so long as there is at least one nucleotide in common
# between two bases, they are considered to match
permissive_matches = {
        # Adenine
        'A': {'A', 'D', 'H', 'M', 'N', 'R', 'V', 'W'},
        # Cytosine
        'C': {'B', 'C', 'H', 'M', 'N', 'S', 'V', 'Y'},
        # Guanine
        'G': {'B', 'D', 'G', 'K', 'N', 'R', 'S', 'V'},
        # Thymine
        'T': {'B', 'D', 'H', 'K', 'N', 'T', 'W', 'Y'},
        # A or G
        'R': {'A', 'B', 'D', 'G', 'H', 'K', 'M', 'N', 'R', 'S', 'V', 'W'},
        # C or T
        'Y': {'B', 'C', 'D', 'H', 'K', 'M', 'N', 'S', 'T', 'V', 'W', 'Y'},
        # G or C
        'S': {'B', 'C', 'D', 'G', 'H', 'K', 'M', 'N', 'R', 'S', 'V', 'Y'},
        # A or T
        'W': {'A', 'B', 'D', 'H', 'K', 'M', 'N', 'R', 'T', 'V', 'W', 'Y'},
        # G or T
        'K': {'B', 'D', 'G', 'H', 'K', 'N', 'R', 'S', 'T', 'V', 'W', 'Y'},
        # A or C
        'M': {'A', 'B', 'C', 'D', 'H', 'M', 'N', 'R', 'S', 'V', 'W', 'Y'},
        # C or G or T
        'B': {'B', 'C', 'D', 'G', 'H', 'K', 'M', 'N', 'R', 'S', 'T', 'V', 'W',
              'Y'},
        # A or G or T
        'D': {'A', 'B', 'D', 'G', 'H', 'K', 'M', 'N', 'R', 'S', 'T', 'V', 'W',
              'Y'},
        # A or C or T
        'H': {'A', 'B', 'C', 'D', 'H', 'K', 'M', 'N', 'R', 'S', 'T', 'V', 'W',
              'Y'},
        # A or C or G
        'V': {'A', 'B', 'C', 'D', 'G', 'H', 'K', 'M', 'N', 'R', 'S', 'V', 'W',
              'Y'},
        # A or C or G or T
        'N': {'A', 'B', 'C', 'D', 'G', 'H', 'K', 'M', 'N', 'R', 'S', 'T', 'V',
              'W', 'Y'},
}


file_names_style = {
    # g1, genotype; g2, region
    'style1': r'(B3|D4|D8).*?(N450_MFNCR|N450|MFNCR)',
    # g1, dataset; g2, model settings (for BEAST runs)
    'style2': r'([^-]+)-([A-Z])\..*',
    # g1, genotype; g2, dataset; g3, region
    'style3': r'(B3|D4|D8)_([A-Z_]*)(?:_|-).*?(N450_MFNCR|N450|MFNCR)'
}


def nucleotide_match(base1, base2, partial_matches_allowed=True):
    """
    Compares two bases and returns True if there is a match and False
    otherwise.

    Parameters
    ----------
    base1, base2 : str
        A string of len 1 representing a nucleotide.
    partial_matches_allowed : bool
        True if, for example, R and A or G (and vice-versa) should be
        considered a match, False if not.

    Returns
    -------
    bool
        True if:
        * identical strs
        * partial_matches_allowed and base1 and base2 are a permitted match
        False if:
        * different bases and base2 is not in the permitted matches for base1
    """
    base1_upper = base1.upper()
    base2_upper = base2.upper()
    for base in [base1_upper, base2_upper]:
        if base not in [*permissive_matches.keys(), '-']:
            raise ValueError(f'base ({base}) character not known IUPAC symbol')
    return (base1_upper == base2_upper or
            (partial_matches_allowed and
             base2_upper in permissive_matches.get(base1_upper, set())))


def num_diff_seq_pair(seq1, seq2, partial_matches_allowed=True):
    """
    Takes in two aligned sequences (with '-') and returns absolute number of
    differences between them. If partial_matches_allowed, ambiguous bases will
    considered a match to any of the nucleotides they represent (i.e., 'R' will
    be counted as match to either 'A' or 'G' and 'A'/'G' will be counted as a
    match to 'R'). Counts gaps and difference in length between sequences as
    differences.
    """
    # add one each time sequences of each pair differ
    # adds the difference in length of the sequences
    return sum(
            1 for a, b in zip(seq1, seq2)
            if not nucleotide_match(a, b, partial_matches_allowed)
            ) + abs(len(seq1) - len(seq2))


# for editable text within Inkscape
rcParams['svg.fonttype'] = 'none'


def year_fraction_from_id(identifier):
    """
    Takes sample record id like 'H123456789_2017.1489' and returns year
    fraction (float following '_').
    """
    return float(identifier.split('_')[-1])


def sample_pairs_times(tree_path, tree_type='nexus'):
    """
    Given a Bayesian-inferred phylogenetic tree `tree_path` of type tree_type
    returns a `pandas.DataFrame` containing time information for each pair of
    samples in the tree.

    Parameters
    ----------
    tree_path : str
        A path to a tree file with trip tip labels as "SampleID_year.float".
    tree_type : str, default 'nexus'
        The tree-encoding format. Any permitted by Dendropy's `get_from_path`
        method.

    Returns
    -------
    pd.DataFrame
        Pandas data frame containing the time information gathered from the
        tree file for each pair of samples.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks between the pair of samples (int)
        DtMRCA: weeks between most recent sample and the Bayesian-predicted
            MRCA (float)
        tMRCA: time of Bayesian-predicted MRCA in weeks (float)
    """
    # read tree from file
    tree = ddp.Tree().get_from_path(tree_path, tree_type,
                                    preserve_underscores=True)
    # produce tree distance matrix
    distance_matrix = tree.node_distance_matrix()

    # get pairs of sample nodes
    node_pairs = list(combinations(tree.leaf_nodes(), 2))

    # data for plotting and validation
    pairs_data = []

    for (s1_node, s2_node) in node_pairs:
        # for (sample1, sample2) in sample_pairs:
        # (year+wk/52)*52 is int
        sample1 = s1_node.taxon.label
        sample2 = s2_node.taxon.label

        time1 = int(year_fraction_from_id(sample1) * 52)
        time2 = int(year_fraction_from_id(sample2) * 52)
        # time between samples in weeks (originally in years)
        pair_delta = abs(time1 - time2)

        # get time of the MRCA as calculated by BEAST
        # * find MRCA node
        s1_s2_mrca = distance_matrix.mrca(s1_node, s2_node)
        # * calculate the distance of each node from the root (cannot use
        #   distance_from_tip as this calculates time between node and most
        #   recent child of the node (not most recent tip in the tree);
        #   tree distances in years, so converted to weeks
        node_root_dist = {node: node.distance_from_root() * 52
                          for node in [s1_node, s2_node, s1_s2_mrca]}
        most_recent_sample_node = max(node_root_dist,
                                      key=lambda node: node_root_dist[node])
        # * number of weeks between the most recent sample and the MRCA
        Dt_mrca = (node_root_dist[most_recent_sample_node] -
                   node_root_dist[s1_s2_mrca])
        # * get time of MRCA (may not be needed)
        t_mrca = max(time1, time2) - Dt_mrca

        # add info to list of data
        pairs_data.append(
                [sample1, sample2, time1, time2, pair_delta, Dt_mrca, t_mrca])
    return pd.DataFrame(
            columns='Sample1, Sample2, ts1, ts2, Dt, DtMRCA, '
                    'tMRCA'.split(', '),
            data=pairs_data)


def sample_pairs_distances(pair_data, fasta_path):
    """
    Calculates the pairwise distances for the samples in the `pair_data`
    pandas.DataFrame from the sequences in the `fasta_path`.

    Parameters
    ----------
    pairs_info : pd.DataFrame
        Pandas data frame containing the time information for each pair of
        samples gathered from a tree file.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
    fasta_path : str
        Path to the fasta file to analyse.

    Returns
    -------
    pd.DataFrame
        Pandas data frame like the input one, with an extra column for the
        Hamming distance calculated for each sample pair.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
    """
    record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))

    def get_distance(row):
        """
        Calculates distance between sequences of the two samples in the df row.
        Ambiguous bases and gaps are counted as differences.
        """
        seq1 = record_dict[row['Sample1']].seq
        seq2 = record_dict[row['Sample2']].seq
        return num_diff_seq_pair(seq1, seq2, partial_matches_allowed=True)

    pair_data['Dist'] = pair_data.apply(get_distance, axis=1)
    return pair_data


def poisson_mode_and_alpha(expected, alpha):
    """
    Returns mode number of expected substitutions and upper & lower limits
    within which falls `alpha` fraction of the occurrences for a poisson
    distribution with lambda=`expected` value.

    Parameters
    ----------
    expected : float
        Expected substitutions.
    alpha : float
        Probability interval containing alpha fraction of the Poisson
        distribution.

    Returns
    -------
    tuple of int
        mode: Typical number of substitutions expected.
        lower_limit, upper_limit: Upper and lower number of substitutions
            between which alpha fraction of the Poisson distribution is
            contained.
    """
    mean, var = poisson.stats(expected, moments='mv')
    mode = math.floor(mean)  # round down to closest integer
    lower_limit, upper_limit = poisson.interval(alpha, expected, loc=0)

    return mode, lower_limit, upper_limit


def expected_substitutions(rate, sites, alpha, time_range):
    """
    Given a substitution `rate`, the number of `sites`, the `alpha`probability
    interval `alpha` and the `time_range` to use returns expected, lower and
    upper values for the expected substitutions for a Poisson distribution with
    lambda=rate*time.

    Parameters
    ----------
    rate : float
        Substitution rate in substitutions/(site*year).
    sites : int
        Number of sites in the genomic region to which the rate applies.
    alpha : float
        Fraction of the Poisson distribution to fall within interval (0-1).
        e.g., 0.95: 95% of the Poisson distribution at a given time will fall
        within the lower-upper range of number of substitutions.
    time_range : tuple
        Time range for which expected values are calculated (ints or floats),
        (min_time, max_time)

    Returns
    -------
    expected_data : pd.DataFrame
        Pandas dataframe containing expected substitution range for each time
        in the interval given.
        Time: time for which mode, lower and upper are calculated, in weeks
            (int)
        Typical: typical number of substitutions expected at each time (int)
        Lower: lower # of substitutions at the confidence (alpha) interval
            given (int)
        Upper: upper # of substitutions for the confidence (alpha) interval
            given (int)
    """
    min_time, max_time = time_range
    expected_data = pd.DataFrame({'Time': range(math.floor(min_time),
                                                math.ceil(max_time) + 1)})

    def poisson_values(time):
        """Expected substitutions for time with rate (poisson lambda)."""
        # rate in subs/(site.year), convert to subs/week; multiply by each time
        # point to obtain expected substitutions at that time
        poisson_rate = (rate / 52) * sites * time
        mode, lower, upper = poisson_mode_and_alpha(poisson_rate, alpha)
        return {'Typical': mode, 'Lower': lower, 'Upper': upper}

    expected_data[['Typical', 'Lower', 'Upper']] = expected_data['Time'].apply(
            lambda t: pd.Series(poisson_values(t)))
    return expected_data


def get_cumulative_evolution_times(pairs_data, relative_times_to_ca,
                                   possible_margin=0.1):
    """
    Calculate the time each pair of samples had to evolve since the presumed
    common ancestor for pairs of samples collected within a smaller time frame
    than that between the most recent sample and the presumed ancestor.

    Parameters
    ----------
    pairs_data : pd.DataFrame
        Pandas data frame containing the information for each pair of samples
        gathered from fasta and tree files.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: number of differences between Sample1 and Sample2 sequences (int)
    relative_times_to_ca : iter
        An iterable of times from a putative common ancestor (DtCA; int) for
        which model is being assessed.
    possible_margin : float, default 0.1
        The level of tolerance for the MRCA time cutoff. 0.1 means that when
        the MRCA of two samples has a time larger than the threshold by up to
        10%, it is considered possible that the MRCA happened within the time
        considered (accounting for errors in the BEAST summary tree or sample
        dates).

    Returns
    -------
    pairs_to_classify : pd.DataFrame
        Pandas data frame with each pair info line repeated for each relative
        time to common ancestor (ignoring sample pairs spanning a period larger
        than that between the most recent sample and the presumed ancestor.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
        DtCA: times since a presumed CA (float)
        max_DtCA: DtCA * (1 + possible_margin) (float)
        tCE: cumulative evolution time (tCE) for the pair since the tCA (float)
    """
    pairs_to_classify = pd.DataFrame(
        columns='Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist, '
                'DtCA, max_DtCA, tCE'.split(', '))
    for relative_t_to_ca in relative_times_to_ca:
        pairs_info_for_DtCA = pairs_data.copy()
        pairs_info_for_DtCA['DtCA'] = relative_t_to_ca
        pairs_info_for_DtCA['max_DtCA'] = (
                relative_t_to_ca * (1 + possible_margin))

        # relative_t_to_ca is the number of weeks from most recent sample
        # to the presumed common ancestor (DtCA)
        # if the difference in time between samples 1 and 2 is longer than the
        # time predicted for the common ancestor, ignore the pair, e.g.:
        #    /--- Sample 1
        #    \--------- Sample 2
        #         |<-->| DtCA     would not make sense to consider these pairs
        pairs_to_consider = pairs_info_for_DtCA[
                pairs_info_for_DtCA['Dt'] <= relative_t_to_ca].reset_index(
                drop=True)
        # calculate cumulative evolution time (tCE)
        # 2 samples may have accumulated differences over the sum of times
        # between each sample and the putative common ancestor
        pairs_to_consider['tCE'] = (
                2 * relative_t_to_ca - pairs_to_consider['Dt'])

        pairs_to_classify = pd.concat([pairs_to_classify, pairs_to_consider],
                                      ignore_index=True)
        print(f'Calculated tCEs for DtCA {relative_t_to_ca}.')
    return pairs_to_classify


def observed_and_predicted_for_dataset(
        pairs_info, relative_ts_to_ca, substitution_rate, num_sites,
        possible_margin=0.1, alpha=0.95):
    """
    Given a pandas.DataFrame containing time and distance information for pairs
    of samples, classifies each pair at each tCA as per the Bayesian estimate
    and the Poisson prediction.

    Parameters
    ----------
    pairs_info : pd.DataFrame
        Pandas data frame containing the information for each pair of samples
        gathered from fasta and tree files.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
    relative_ts_to_ca : iter
        An iterable object containing ints or floats, each representing a time
        between the most recent sample and a presumed common ancestor (DtCA).
    substitution_rate : float
        Substitution rate in substitutions/(site.year)
    num_sites : int
        The number of sites in the MSA.
    possible_margin: float; default 0.1
        The level of tolerance for the MRCA time cutoff. 0.1 means that when
        the MRCA of two samples has a time larger than the threshold by up to
        10%, it is considered possible that the MRCA happened within the time
        considered (accounting for errors in the BEAST summary tree or sample
        dates).
    alpha : float, default 0.95
        Fraction of the Poisson distribution to fall within interval. Between
        0 and 1.
        e.g., 0.95: 95% of the Poisson distribution at a given time will fall
        within the lower-upper range of number of substitutions.

    Returns
    -------
    pairs_results : pd.DataFrame
        Pandas data frame containing the observed and predicted results for
        each file.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample time in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
        DtCA: times since a presumed CA (float)
        max_DtCA: DtCA * (1 + possible_margin) (float)
        tCE: cumulative evolution time (tCE) for the pair since the tCA (float)
        Observed: pair category based on time tree - likely, possible or
            unlikely (str)
        Predicted: predicted category based on model - below, typical or above
            (str)
        Result: result classification based on observed and predicted (str)
    poisson_values : pd.DataFrame
        Pandas data frame containing expected substitution range for each time
        in the interval given.
        Time: time for which mode, lower and upper are calculated, in weeks
            (int)
        Typical: typical number of substitutions expected at each time (int)
        Lower: lower # of substitutions at the confidence (alpha) interval
            given (int)
        Upper: upper # of substitutions for the confidence (alpha) interval
            given (int)
    """
    # tCE for each DtCA
    # pairs_results columns:
    #   Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist, DtCA, max_DtCA,
    #   tCE
    pairs_results = get_cumulative_evolution_times(
            pairs_info, relative_ts_to_ca, possible_margin)
    # calculate expected substitution range for the full range of tCE
    min_tce = pairs_results['tCE'].min()
    max_tce = pairs_results['tCE'].max()
    # data frame columns: Time, Typical, Lower, Upper
    poisson_values = expected_substitutions(
            substitution_rate, num_sites, alpha, (min_tce, max_tce))

    # classify based on time tree
    pairs_results['Observed'] = np.nan
    # DtMRCA <= DtCA: likely
    likely_pairs = pairs_results['DtMRCA'] <= pairs_results['DtCA']
    pairs_results.loc[likely_pairs, 'Observed'] = 'likely'
    # DtMRCA > DtCA: unlikely
    unlikely_pairs = pairs_results['DtMRCA'] > pairs_results['max_DtCA']
    pairs_results.loc[unlikely_pairs, 'Observed'] = 'unlikely'
    # remaining (else): possible
    pairs_results.loc[pairs_results['Observed'].isna(),
                      'Observed'] = 'possible'
    print('Classified pairs based on time tree.')

    # classify based on Poisson predictions and model tCE
    # add subs ranges to df
    pairs_results = pairs_results.join(poisson_values.set_index('Time'),
                                       on='tCE')
    pairs_results['Predicted'] = np.nan
    # distance < low end of expected substitution range for tCE: below
    below_pairs = pairs_results['Dist'] < pairs_results['Lower']
    pairs_results.loc[below_pairs, 'Predicted'] = 'below'
    # distance > high end of expected substitution range for tCE: above
    above_pairs = pairs_results['Dist'] > pairs_results['Upper']
    pairs_results.loc[above_pairs, 'Predicted'] = 'above'
    # remaining (else): expected
    pairs_results.loc[pairs_results['Predicted'].isna(),
                      'Predicted'] = 'typical'
    # drop unnecessary columns
    pairs_results.drop(columns='Typical, Lower, Upper'.split(', '),
                       inplace=True)
    print('Classified pairs based on model prediction.')

    # classify result
    obs_likely = pairs_results['Observed'] == 'likely'
    obs_possible = pairs_results['Observed'] == 'possible'
    obs_unlikely = pairs_results['Observed'] == 'unlikely'
    pre_below = pairs_results['Predicted'] == 'below'
    pre_above = pairs_results['Predicted'] == 'above'
    pre_typical = pairs_results['Predicted'] == 'typical'
    correct_pairs = (
            # likely pairs below max exp subs
            obs_likely & (pre_typical | pre_below)) | (
            # unlikely pairs above exp subs
            obs_unlikely & pre_above) | (
            # possible always correct (borderline)
            obs_possible)
    incorrect_pairs = (
            # false positives
            obs_likely & pre_above) | (
            # false negatives
            obs_unlikely & (pre_typical | pre_below))
    pairs_results['Result'] = np.nan
    pairs_results.loc[correct_pairs, 'Result'] = 'correct'
    pairs_results.loc[incorrect_pairs, 'Result'] = 'incorrect'
    print('Classified model results.\n')
    # pairs_results columns:
    #   Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist, DtCA, max_DtCA,
    #   tCE, Observed, Predicted, Result
    # poisson_values columns: Time, Typical, Lower, Upper
    return pairs_results, poisson_values


def observed_and_predicted(
        tree_fastas, relative_times_to_ca, rates, possible_margin=0.1,
        alpha=0.95, tree_encoding='nexus'):
    """
    Given combinations of trees and fasta files, calculates distance tMRCA, and
    cumulative evolutionary time for combinations of pairs of samples for times
    in relative_times_to_ca and analyses observed and predicted results.

    Parameters
    ----------
    tree_fastas : dict
        A dictionary mapping a time-scaled phylogenetic tree to a list of fasta
        files containing MSAs for the region(s) used to infer the tree.
        e.g., 'B3_N450_MFCR.nex': ['B3_N450.fas', 'B3_MFNCR.fas']
    relative_times_to_ca : iter
        An iterable object containing ints or floats, each representing a time
        between the most recent sample and a presumed common ancestor (DtCA).
    rates : pd.DataFrame
        Pandas data frame containing substitution rates.
        Genotype, Region: Genotype and region for which the rate was calculated
            (str)
        Sites: number of sites in the MSA for the region analysed (int)
        Rate: substitution rate in substitutions/(site.year) (float)
    possible_margin : float; default 0.1
        The level of tolerance for the MRCA time cutoff. 0.1 means that when
        the time of the MRCA in the time tree for a pair of samples has a time
        larger than the relative time to a CA being checked by up to 10%, it is
        considered possible that the MRCA happened within the time considered
        (accounting for errors in the BEAST summary tree or sample dates).
    alpha : float
        Fraction of the Poisson distribution to fall within interval. Between
        0 and 1.
        e.g., 0.95: 95% of the Poisson distribution at a given time will fall
        within the lower-upper range of number of substitutions.
    tree_encoding : str
        The format of the tree file, e.g., 'newick', 'nexus'.

    Returns
    -------
    results : pd.DataFrame
        Pandas data frame containing the observed and predicted results for
        all files.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample times in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
        DtCA: times since a presumed CA (float)
        max_DtCA: DtCA * (1 + possible_margin) (float)
        tCE: cumulative evolution time (tCE) for the pair since the tCA (float)
        Observed: pair category based on time tree - likely, possible or
            unlikely (str)
        Predicted: predicted category based on model - below, typical or above
            (str)
        Result: result classification based on observed and predicted (str)
        Genotype, Dataset, Region: Genotype, set of data and region for which
            the rate was calculated (str)
    predicted : pd.DataFrame
        Pandas dataframe containing expected substitution range for each time
        in the interval given for each dataset, genotype and region.
        Time: time for which mode, lower and upper are calculated, in weeks
            (int)
        Typical: typical number of substitutions expected at each time (int)
        Lower: lower # of substitutions at the confidence (alpha) interval
            given (int)
        Upper: upper # of substitutions for the confidence (alpha) interval
            given (int)
        Genotype, Dataset, Region: Genotype, set of data and region for which
            the substitutions were calculated (str)
    """
    results = pd.DataFrame(
            columns='Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist, '
                    'DtCA, max_DtCA, tCE, Observed, Predicted, Result, '
                    'Genotype, Dataset, Region'.split(', '))
    predicted = pd.DataFrame(
            columns='Time, Typical, Lower, Upper, Genotype, Dataset, '
                    'Region'.split(', '))
    in_dir = None
    num_trees = len(tree_fastas)
    trees_processed = 1
    for tree_path, fastas in tree_fastas.items():
        print(f'\nProcessing tree {trees_processed}/{num_trees}\n{tree_path}')
        # get tree info
        # pairs_time_info columns:
        #   Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA
        pairs_time_info = sample_pairs_times(
                tree_path, tree_type=tree_encoding)
        for fasta_path in fastas:
            print(f'Processing sequences in {fasta_path}')
            # obtain information about the data
            base_dir, filename = os.path.split(fasta_path)
            if not in_dir:
                in_dir = base_dir
            basename, file_ext = os.path.splitext(filename)
            match = re.search(file_names_style['style3'], basename)
            genotype = match.group(1)
            dataset = match.group(2).replace('_', ' ')
            region = match.group(3).replace(
                    'N450', 'N-450').replace('MFNCR', 'MF-NCR')

            # get distances
            # pairs_info columns:
            #   Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist
            pairs_info = sample_pairs_distances(pairs_time_info, fasta_path)
            print('Obtained time and distance information for pairs in tree.')
            # rate and number of sites in MSA from rates.csv
            rate = rates.loc[(genotype, region), 'Rate']
            num_sites = rates.loc[(genotype, region), 'Sites']
            # get observed and predicted values
            # dataset_results columns:
            #   Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist, DtCA,
            #   max_DtCA, tCE, Observed, Predicted, Result
            # poisson_values columns:
            #   Time, Mode, Lower, Upper
            (dataset_results,
             poisson_values) = observed_and_predicted_for_dataset(
                    pairs_info, relative_times_to_ca, rate, num_sites,
                    possible_margin=possible_margin, alpha=alpha)

            dataset_results['Genotype'] = genotype
            dataset_results['Dataset'] = dataset
            dataset_results['Region'] = region

            poisson_values['Genotype'] = genotype
            poisson_values['Dataset'] = dataset
            poisson_values['Region'] = region
            # results columns:
            #   Sample1, Sample2, ts1, ts2, Dt, DtMRCA, tMRCA, Dist, DtCA,
            #   max_DtCA, tCE, Observed, Predicted, Result, Genotype, Dataset,
            #   Region
            results = pd.concat([results, dataset_results],
                                ignore_index=True)
            # predicted columns:
            #   Time, Typical, Lower, Upper, Genotype, Dataset, Region
            predicted = pd.concat([predicted, poisson_values],
                                  ignore_index=True)
        trees_processed += 1

    out_dir = in_dir.replace('input', 'output')
    out_results = os.path.join(out_dir, 'pair_results.csv')
    results.to_csv(out_results, index=False)
    out_predicted = os.path.join(out_dir, 'predicted.csv')
    predicted.to_csv(out_predicted, index=False)
    print(f'Completed processing trees and fasta files. Results in '
          f'{out_results}')
    return results, predicted


def get_axes_mid_points(figure):
    """
    Given a matplotlib figure containing one or multiple sets of axes, returns
    the horizontal and vertical mid points of the axes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        A matplotlib figure object containing one or multiple axes.

    Returns
    -------
    (x_centre, y_centre) : tuple
        Length 2 tuple containing two floats. The first is the mid point of the
        x axes and the second is the mid point of the y axes.
    """
    # positions of the first and last subplots
    top_left_subplot_pos = figure.get_axes()[0].get_position()
    bottom_right_subplot_pos = figure.get_axes()[-1].get_position()
    # x and y limits of the subplots
    x_min = top_left_subplot_pos.xmin
    x_max = bottom_right_subplot_pos.xmax
    y_max = top_left_subplot_pos.ymax
    y_min = bottom_right_subplot_pos.ymin
    x_centre = x_min + (x_max - x_min) / 2
    y_centre = y_min + (y_max - y_min) / 2
    return x_centre, y_centre


def add_axes_labels_to_figure(figure, x_label='', y_label=''):
    """
    Takes in a matplotlib figure containing one or multiple axes and returns
    the figure with the requested x/y labels added at the centre of the set of
    axes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        A matplotlib figure object.
    x_label : str, default ''
        The text to use to label the x axes. If an empty string, no horizontal
        text is added.
    y_label : str, default ''
        The text to use to label the y axes. If an empty string, no vertical
        text is added.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure object with the centered labels added.
    """
    (x_centre, y_centre) = get_axes_mid_points(figure)
    # add labels centered to the ax(es)
    if x_label:
        figure.text(x=x_centre, y=0, s=x_label,
                    ha='center', weight='bold')
    if y_label:
        figure.text(x=0, y=y_centre, s=y_label,
                    va='center', rotation='vertical', weight='bold')
    return figure


def make_plots(results, expected, out_dir,
               sets_to_analyse=tuple(), times_to_analyse=tuple(),
               alpha=0.95, sns_style='ticks', sns_context='paper',
               save_img_as=('.png',)):
    """
    Plots results from a model predicting sequence pair relatedness.

    Parameters
    ----------
    results : pd.DataFrame
        Pandas data frame containing observed and predicted results for
        multiple trees and regions.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample times in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
        DtCA: times since a presumed CA (float)
        max_DtCA: DtCA * (1 + possible_margin) (float)
        tCE: cumulative evolution time (tCE) for the pair since the tCA (float)
        Observed: pair category based on time tree - likely, possible or
            unlikely (str)
        Predicted: predicted category based on model - below, typical or above
            (str)
        Result: result classification based on observed and predicted (str)
        Genotype, Dataset, Region: Genotype, set of data and region for which
            the rate was calculated (str)
    expected : pd.DataFrame
        Pandas dataframe containing expected substitution range for each time
        in the interval given for each dataset, genotype and region.
        Time: time for which mode, lower and upper are calculated, in weeks
            (int)
        Typical: typical number of substitutions expected at each time (int)
        Lower: lower # of substitutions at the confidence (alpha) interval
            given (int)
        Upper: upper # of substitutions for the confidence (alpha) interval
            given (int)
        Genotype, Dataset, Region: Genotype, set of data and region for which
            the substitutions were calculated (str)
    out_dir : str
        Path to a directory where figures will be saved.
    sets_to_analyse : tuple, default tuple()
        The data sets to plot as entered in the results df 'Datasets' column.
    times_to_analyse : tuple or list, default tuple()
        A list or tuple containing the times (int) for which each dataset is
        plotted.
    alpha : float, default 0.95
        Fraction of the Poisson distribution to fall within interval. Between
        0 and 1. 0.95: 95% of the Poisson distribution at a given time will
        fall within the lower-upper range of number of substitutions.
    sns_style : str, default 'ticks'
        Argument passed to seaborn set_style method, which adjusts plot
        settings. Any permitted bt sns.set_style.
    sns_context : str, default 'paper'
        Argument passed to seaborn set_context method, which adjusts plot
        settings. Any permitted bt sns.set_context.
    save_img_as : tuple, default ('.png')
        Tuple containing figure formats to save matplotlib plot as. Any file
        type supported by matplolib.

    Returns
    -------
    None
    """
    # slice df based on the times to analyse
    if times_to_analyse:
        subset = results[results['DtCA'].isin(times_to_analyse)
                ].reset_index(drop=True)
        ts_to_ca = times_to_analyse
    # use full df if no DtCA subset is given
    else:
        subset = results
        ts_to_ca = results['DtCA'].unique()

    # if no data set subset given, analyse all data sets
    if not sets_to_analyse:
        sets_to_analyse = results['Dataset'].unique()

    # set up the plot style
    sns.set_style(sns_style)
    sns.set_context(sns_context)
    typical_label = 'typical'
    exp_range_label = f'{alpha} of Poisson'
    below_exp_label = 'over-corrected'

    for dataset in sets_to_analyse:
        ds_subset = subset[subset['Dataset'] == dataset].reset_index(drop=True)
        dse_subset = expected[expected['Dataset'] == dataset
                ].reset_index(drop=True)
        for genotype in ds_subset['Genotype'].unique():
            gen_subset = ds_subset[ds_subset['Genotype'] == genotype
                    ].reset_index(drop=True)
            ge_subset =  dse_subset[dse_subset['Genotype'] == genotype
                    ].reset_index(drop=True)
            for t_to_ca in ts_to_ca:
                t_subset = gen_subset[gen_subset['DtCA'] == t_to_ca
                        ].reset_index(drop=True)
                (unlikely_label, possible_label, likely_label) = (
                    'unlikely', 'possible', 'likely')
                # set up facet grid
                num_regions = ge_subset['Region'].nunique()
                facet_grid_args = dict(
                        hue='Observed', hue_order=[unlikely_label,
                        possible_label, likely_label],
                        hue_kws=dict(marker=['^', '.', 'v']),
                        palette={unlikely_label: '#7570b3',
                                 possible_label: '#d95f02',
                                 likely_label: '#1b9e77'},
                        height=2.5, aspect=1.3, despine=True,
                        sharex=True, sharey=False, margin_titles=True)
                if num_regions > 1:
                    facet_grid_args.update(dict(
                            row='Region', row_order=['N-450', 'MF-NCR']))
                g = sns.FacetGrid(t_subset, **facet_grid_args)
                g.map(plt.scatter, 'tCE', 'Dist', zorder=4)
                region_order = ['N-450', 'MF-NCR']
                regions = [reg
                           for reg in region_order
                           if reg in ge_subset['Region'].unique()]

                # for each facet, add the expected substitutions line and range
                for i in range(len(regions)):
                    ax = g.axes.flat[i]
                    region = regions[i]
                    xmin, xmax, ymin, ymax = ax.axis()
                    # adjust axis limits
                    tmin = max(math.floor(xmin), 2)
                    tmax = math.ceil(xmax)
                    smax = max(ymax, 1)
                    ax.set(xlim=(tmin, tmax), ylim=(0, smax))
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # get the relevant expected substitutions for the facet
                    exp = ge_subset[(
                            ge_subset['Time'].between(tmin, tmax)) & (
                            ge_subset['Region'] == region)
                            ].reset_index(drop=True)
                    tce_vals = exp['Time'].values
                    typical_vals = exp['Typical'].values
                    upper_limit = exp['Upper'].values
                    lower_limit = exp['Lower'].values
                    # plot typical values
                    ax.plot(tce_vals, typical_vals,
                            linewidth=3, color='#4575b4', zorder=3,
                            label=typical_label)
                    # plot expected range
                    ax.fill_between(
                            tce_vals, upper_limit, lower_limit,
                            facecolor='#91bfdb', edgecolor='#91bfdb',
                            alpha=1, zorder=2,
                            label=exp_range_label)
                    # plot over-corrected range if min limit is 1 or more
                    if max(lower_limit) > 0:
                        # plot range below expected substitutions
                        ax.fill_between(
                                tce_vals, lower_limit, 0,
                                facecolor='#D0D0D0', edgecolor='#D0D0D0',
                                alpha=1, zorder=1,
                                label='over-corrected')

                # set up labels
                g.set_axis_labels('', '')
                [plt.setp(ax.texts, text='') for ax in g.axes.flat]
                g.set_titles(row_template='{row_name}')
                g.fig.subplots_adjust(hspace=0.1)
                # add a figure title
                g.fig.text(0, 1,
                        s=fr'{genotype} - {dataset} '
                          fr'$(\Delta t_{{CA}} = {t_to_ca}\ weeks)$',
                        weight='bold')
                g.fig = add_axes_labels_to_figure(
                        g.fig,
                        x_label=r'$\mathbf{\Delta t_{CE}}$ (weeks)',
                        y_label='number of substitutions')
                g.fig.tight_layout()

                # configure legend
                current_handles, current_labels = plt.gca(
                        ).get_legend_handles_labels()

                lab_han = dict(zip(current_labels, current_handles))
                exp_legend_order = [
                        typical_label, exp_range_label, below_exp_label]
                obs_legend_order = [
                        unlikely_label, possible_label, likely_label]

                ordered_exp_lab_han = [(label, lab_han[label])
                                       for label in exp_legend_order
                                       if label in lab_han]
                exp_ordered_labels, exp_ordered_handles = list(
                        zip(*ordered_exp_lab_han))
                ordered_obs_lab_han = [(label, lab_han[label])
                                       for label in obs_legend_order
                                       if label in lab_han]
                obs_ordered_labels, obs_ordered_handles = list(
                        zip(*ordered_obs_lab_han))

                # Create a legend for the expected series
                exp_legend = plt.legend(
                        handles=exp_ordered_handles, labels=exp_ordered_labels,
                        bbox_to_anchor=(0.57, 0), loc='upper right', ncol=1,
                        bbox_transform=g.fig.transFigure, frameon=False)

                # Add the legend manually to the current Axes.
                plt.gca().add_artist(exp_legend)

                # Create another legend for the observed series
                plt.legend(
                        handles=obs_ordered_handles, labels=obs_ordered_labels,
                        bbox_to_anchor=(0.57, 0), loc='upper left', ncol=1,
                        bbox_transform=g.fig.transFigure, frameon=False)

                # save image
                for fig_type in save_img_as:
                    fig_ext = (f'.{fig_type}'
                               if fig_type[0] != '.' else fig_type)
                    file_name = f'{genotype}-{dataset}-{t_to_ca:03d}'.replace(
                            ' ', '_')
                    g.savefig(os.path.join(out_dir, f'{file_name}{fig_ext}'),
                              dpi=300)
                plt.close()
                print(f'Plot produced for {genotype} {dataset} '
                      f'DtCA={t_to_ca:03d} weeks.')


# --------------------------------------------------------------------------- #
#                              Ignoring possibles
# --------------------------------------------------------------------------- #

# Ignoring possibles because tMRCA too close to tMRKCA to know whether null
# hypothesis is True or False
#
#                       Predicted
#                         a   | r
#                       B   E   A
# Observed  L/H0 true   TN  TN  FP
#           P/H0 u/k    -   -   -   ignored
#           U/H0 false  FN  FN  TP
#
# a: H0 accepted        BEAST tMRCA<=DtCA:      Model pred subs:
# r: H0 rejected        L: Likely               B: below exp range
#                       P: Possible             E: in exp range / typical
#                       U: Unlikely             A: above exp range
#
# H0: Two sequences are related within the time frame given
#
# FP = type I error = true H0 rejected,
#      * seqs related, but predicted as unrelated
#      * likely (L) in BEAST, but above (A) expected #subs in model - LA
#      * the error to avoid here
# FN = type II error = false H0 non-rejected
#      * seqs unrelated, but predicted as related
#      * unlikely (U) in BEAST, but below (B) or within the expected range (E)
#        of #subs in model - UB + UE
# TP = false H0 rejected
#      * seqs not related and predicted as not related
#      * unlikely (U) in BEAST and above (A) expected #subs in model - UA
# TN = true H0 accepted
#      * seqs related and predicted as related
#      * likely (L) in BEAST and below (B) or within the expected range (E)
#        of #subs in model - LB + LE


def model_stats_excluding_possibles(
        results, calculate_against, group_by=('Genotype', 'Dataset', 'Region'),
        bin_size=4):
    """
    Calculates true positive, true negative, false positive and false negative
    rates for the summary of predictions for each DtCA, genotype and genome
    region. Excludes possible sample pairs from analysis.

    Parameters
    ----------
    results : pd.DataFrame
        Pandas data frame containing observed and predicted results for
        multiple trees and regions.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample times in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
        DtCA: times since a presumed CA (float)
        max_DtCA: DtCA * (1 + possible_margin) (float)
        tCE: cumulative evolution time (tCE) for the pair since the tCA (float)
        Observed: pair category based on time tree - likely, possible or
            unlikely (str)
        Predicted: predicted category based on model - below, typical or above
            (str)
        Result: result classification based on observed and predicted (str)
        Genotype, Dataset, Region: Genotype, set of data and region for which
            the rate was calculated (str)
    calculate_against : str
        Heading of the data frame column by which results are grouped for the
        rate calculation.
    group_by : list or tuple or str
        Columns in the data frame by which the results will be aggregated.
    bin_size : int or None, default 4
        Size of the bins to cut the aggregate column into. If None, no bins
        will be used.

    Returns
    -------
    pd.DataFrame
        Genotype, Dataset, Region: the set of data for which the results were
            calculated (str)
        calculate_against str: the column values of the initial dataframe
            column by which the results were aggregated (int)
        Rates: the type of rate that the 'Rate' column value refers to (str)
        Rate: true/false positive/negative rate values (%)
    """
    columns_needed = [calculate_against, 'Observed', 'Result']
    if type(group_by) in [tuple, list]:
        columns_needed.extend(group_by)
    elif type(group_by) == str:
        columns_needed.append(group_by)
    else:
        raise TypeError(f'Group by should be a tuple, list or string, but it '
                        f'is {type(group_by)}.')

    smaller_df = results[columns_needed].reset_index(drop=True)
    aggregate_col_pos = columns_needed.index(calculate_against)

    if bin_size:
        calculate_against_bins_col = f'{calculate_against}_bins'
        min_aggregate_value = smaller_df[calculate_against].min()
        max_aggregate_value = smaller_df[calculate_against].max()
        categories = list(
            range(min_aggregate_value, max_aggregate_value + 1, bin_size))
        smaller_df[calculate_against_bins_col] = pd.cut(
            smaller_df[calculate_against],
            bins=categories, labels=categories[1:],
            # to label with intervals
            # [f"{'(' if i > 0 else '['}"  # lowest limit included label
            #  f"{int(categories[i])}-{int(categories[i + 1])}]"
            #  for i in range(len(categories) - 1)]
            include_lowest=True)
        # replace calculate_against col in the needed cols list with the col
        # containing the bins
        columns_needed[aggregate_col_pos] = calculate_against_bins_col

    # count the number of sample pairs in each category of Observed/Result
    # values
    results_pt = pd.pivot_table(
            smaller_df[columns_needed],
            columns=['Observed', 'Result'],
            index=[*group_by, columns_needed[aggregate_col_pos]],
            aggfunc=len, fill_value=0, margins=True, margins_name='Total')

    results_pt['Total_ex_Possible'] = (
            results_pt['Total'] - results_pt.loc[:, ('possible', 'correct')])
    likely_correct = results_pt.loc[:, ('likely', 'correct')]
    likely_incorrect = results_pt.loc[:, ('likely', 'incorrect')]
    total_likely = likely_correct + likely_incorrect
    unlikely_correct = results_pt.loc[:, ('unlikely', 'correct')]
    unlikely_incorrect = results_pt.loc[:, ('unlikely', 'incorrect')]
    total_unlikely = unlikely_correct + unlikely_incorrect
    results_pt['TN'] = 100 * (likely_correct / total_likely)
    results_pt['TP'] = 100 * (unlikely_correct / total_unlikely)
    results_pt['FN'] = 100 * (unlikely_incorrect / total_unlikely)
    results_pt['FP'] = 100 * (likely_incorrect / total_likely)

    # Total row not needed
    results_pt.drop(['Total'], inplace=True)
    # transfer pivot table top column index into the row index
    res_pt_stack = results_pt['TN, TP, FN, FP'.split(', ')].stack()

    return pd.melt(
            res_pt_stack.reset_index(),
            id_vars=[*group_by, columns_needed[aggregate_col_pos]],
            value_vars='TN, TP, FN, FP'.split(', '),
            var_name='Rates', value_name='Rate')


# --------------------------------------------------------------------------- #
#                              Including possibles
# --------------------------------------------------------------------------- #
#
# Including possibles as being true whether  null hypothesis is accepted or not
#
#                       Predicted
#                         a   | r
#                       B   E   A
# Observed  L/H0 true   TN  TN  FP
#           P/H0 u/k    TN  TN  TP
#           U/H0 false  FN  FN  TP
#
# a: H0 accepted        BEAST tMRCA<=DtCA:      Model pred subs:
# r: H0 rejected        L: Likely               B: below exp range
#                       P: Possible             E: in exp range / typical
#                       U: Unlikely             A: above exp range
#
# H0: Two sequences are related within the time frame given
#
# FP = type I error = true H0 rejected,
#      * seqs related, but predicted as unrelated
#      * likely (L) in BEAST, but above (A) expected #subs in model - LA
#      * the error to avoid here
# FN = type II error = false H0 non-rejected
#      * seqs unrelated, but predicted as related
#      * unlikely (U) in BEAST, but below (B) or within the expected range (E)
#        of #subs in model - UB + UE
# TP = false H0 rejected
#      * seqs not related and predicted as not related
#      * unlikely (U) in BEAST and above (A) expected #subs in model or
#      possible (P) and A - UA + PA
# TN = true H0 accepted
#      * seqs related and predicted as related
#      * likely (L) in BEAST and below (B) or within the expected range (E)
#        of #subs in model or possible (P) and B or E - LB + LE + PB + PE


def model_stats_including_possibles(results):
    """
    Calculates true positive, true negative, false positive and false negative
    rates for the summary of predictions for each DtCA, genotype and genome
    region. Assumes result for possible sample pairs are always correct.

    Parameters
    ----------
    results : pd.DataFrame
        Pandas data frame containing observed and predicted results for
        multiple trees and regions.
        Sample1, Sample2: sample ids (str)
        ts1, ts2: sample times in weeks (float)
        Dt: weeks difference between the pair of samples (int)
        DtMRCA: weeks between most recent sample and MRCA (float)
        tMRCA: time of MRCA in weeks (float)
        Dist: Hamming distance between Sample1 and Sample2 (int)
        DtCA: times since a presumed CA (float)
        max_DtCA: DtCA * (1 + possible_margin) (float)
        tCE: cumulative evolution time (tCE) for the pair since the tCA (float)
        Observed: pair category based on time tree - likely, possible or
            unlikely (str)
        Predicted: predicted category based on model - below, typical or above
            (str)
        Result: result classification based on observed and predicted (str)
        Genotype, Dataset, Region: Genotype, set of data and region for which
            the rate was calculated (str)

    Returns
    -------
    pd.DataFrame
        Genotype, Dataset, Region: the set of data for which the results were
            calculated (str)
        DtCA: the time between the most recent sample and a presumed common
            ancestor (int)
        Rates: the type of rate that the 'Rate' column value refers to (str)
        Rate: true/false positive/negative rate percentages (float)
    """
    relevant_cols = results[[
            'DtCA', 'Observed', 'Predicted', 'Genotype', 'Dataset', 'Region']]

    results_pt = pd.pivot_table(
            relevant_cols,
            columns=['Observed', 'Predicted'],
            index=['Genotype', 'Dataset', 'Region', 'DtCA'],
            aggfunc=len, fill_value=0)

    # get predicted category columns for each observed category
    preds_for_obs = {}
    for obs, pred in results_pt.columns:
        if obs in ['likely', 'possible', 'unlikely']:
            preds_for_obs.setdefault(obs, [])
            preds_for_obs[obs].append((obs, pred))
    # sums of each observed category
    for obs_col, preds_cols in preds_for_obs.items():
        sum_obs_col_name = f'Sum {obs_col}'
        results_pt[sum_obs_col_name] = 0
        for pred in preds_cols:
            results_pt[sum_obs_col_name] += results_pt[pred]
    # define numerator for each rate
    rates_numerator_cols = dict(
            TP=[('possible', 'above'), ('unlikely', 'above')],
            TN=[('likely', 'below'), ('likely', 'typical'),
                ('possible', 'below'), ('possible', 'typical')],
            FN=[('unlikely', 'below'), ('unlikely', 'typical')],
            FP=[('likely', 'above')])
    # define denominator for each rate
    rates_denominator_cols = dict(
            TP=['Sum possible', 'Sum unlikely'],
            TN=['Sum likely', 'Sum possible'],
            FN=['Sum unlikely'],
            FP=['Sum likely'])

    # calculate rates
    for rate, numerators in rates_numerator_cols.items():
        numerator = 0
        denominator = 0
        for obs_pred_col in numerators:
            if obs_pred_col in results_pt.columns:
                numerator += results_pt[obs_pred_col]
        denominators = rates_denominator_cols[rate]
        for sum_col in denominators:
            denominator += results_pt[sum_col]
        results_pt[rate] = 100 * numerator / denominator

    # Select relevant columns and remove empty column indices
    results_pt = results_pt[[('TN', ''), ('TP', ''),
                             ('FN', ''), ('FP', '')]]
    results_pt.columns = results_pt.columns.droplevel('Predicted')

    # put row indices into columns
    results_pt.reset_index(inplace=True)

    return pd.melt(
            results_pt,
            id_vars='Genotype, Dataset, Region, DtCA'.split(', '),
            value_vars='TN, TP, FN, FP'.split(', '),
            var_name='Rates', value_name='Rate')


def plot_model_stats(
        model_stats, out_handle, x_axis, x_axis_label=None, sns_style='ticks',
        sns_context='paper', img_formats=('.png', '.svg')):
    """
    Given data frame containing true/false positive/negative rates for each
    genotype, region, and relative t to a CA, produces images with facet grid
    plots in the formats required.

    Parameters
    ----------
    model_stats : pd.DataFrame
        Data frame containing model statistics data.
        Genotype, Dataset, Region: the set of data for which the results were
            calculated (str)
        DtCA: the time between the most recent sample and a presumed common
            ancestor (int)
        Rates: the type of rate that the 'Rate' column value refers to (str)
        Rate: true/false positive/negative rate percentages (float)
    out_handle : str
        The path for the files to be saved in, excluding image extension.
    x_axis : str
        Heading of the pandas data frame column where the values for the
        independent variable are found.
    x_axis_label : str or None, default None
        The label to use for the independent variable axis. Given by x_axis if
        None.
    sns_style : str, default 'ticks'
        Argument passed to seaborn set_style method, which adjusts plot
        settings. Any permitted bt sns.set_style.
    sns_context : str, default 'paper'
        Argument passed to seaborn set_context method, which adjusts plot
        settings. Any permitted bt sns.set_context.
    img_formats : tuple, default ('.png', '.svg')
        Tuple containing figure formats to save matplotlib plot as. Any file
        type supported by matplotlib.

    Returns
    -------
    None
    """
    # set up grid args and formatting for legend for different number of
    # regions to be plotted
    grid_args = dict(
            row='Genotype', hue='Rates',
            palette={'FN': '#92c5de', 'TN': '#0571b0',
                     'FP': '#f4a582', 'TP': '#ca0020'},
            margin_titles=True, height=4, aspect=1.5,
            ylim=(0, 100), despine=True)

    titles = dict(row_template='{row_name}')
    subplots_adjustment = dict(hspace=0.1)
    num_regions = model_stats['Region'].nunique()
    if num_regions == 1:
        subplots_adjustment.update(dict(bottom=0.14, left=0.12))
        x_label_pos = (0.49, 0.07)
        y_label_pos = (0.01, 0.55)

    elif num_regions == 2:
        grid_args.update(dict(
                col='Region', col_order='N-450, MF-NCR'.split(', ')))
        titles.update(dict(col_template='{col_name}'))
        subplots_adjustment.update(dict(bottom=0.14, left=0.07))
        x_label_pos = (0.51, 0.06)
        y_label_pos = (0.01, 0.53)

    else:
        grid_args.update(dict(col='Region'))
        titles.update(dict(col_template='{col_name}'))
        subplots_adjustment.update(dict(bottom=0.1, left=0.07))
        x_label_pos = (0.51, 0.05)
        y_label_pos = (0.01, 0.53)

    sns.set_style(sns_style)
    sns.set_context(sns_context)
    grid = sns.FacetGrid(model_stats, **grid_args)
    grid.map(plt.axhline, y=5, linestyle='--', color='lightgrey')
    grid.map(plt.plot, x_axis, 'Rate', linewidth=3)
    grid.set_axis_labels('', '')
    [plt.setp(ax.texts, text='') for ax in grid.axes.flat]
    grid.set_titles(**titles)
    grid.fig.subplots_adjust(**subplots_adjustment)

    grid.fig.text(*x_label_pos, s=x_axis_label if x_axis_label else x_axis,
                  ha='center', va='bottom', weight='bold')
    grid.fig.text(*y_label_pos, s='rate (%)',
                  rotation='vertical', weight='bold',
                  ha='left', va='center')

    # configure legend
    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    lab_han = dict(zip(current_labels, current_handles))
    legend_order = ['TN', 'FN', 'TP', 'FP']

    ordered_lab_han = [(label, lab_han[label])
                       for label in legend_order]
    ordered_labels, ordered_handles = list(zip(*ordered_lab_han))

    x_centre, y_centre = get_axes_mid_points(grid.fig)
    plt.legend(
        handles=ordered_handles, labels=ordered_labels,
        bbox_to_anchor=(x_centre, 0), loc='lower center', ncol=4,
        bbox_transform=grid.fig.transFigure, frameon=False)

    for img_format in img_formats:
        grid.fig.savefig(f'{out_handle}{img_format}', dpi=300)

    plt.close()
    print(f'Model statistics plots saved in {out_handle}')
