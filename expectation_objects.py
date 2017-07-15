import pandas as pd
from itertools import combinations
import numpy as np

from copy import deepcopy
import DimensionalPipeline as dp
from statsmodels.robust.scale import mad




# Overall, the trickiest part of doing density/expectation based clustering was defining what transformation
# of the data to use to calculate expectations with respect to.

class ExpectationsObject():
    # Establish a template for the object. This is mostly a design decision so that it's easier to specify
    # what kinds of objects can be passed into a given method
    def __init__(self):
        self.name = "GenericExpectationsObject"

    def fit(self, df, left_col, right_col):
        # This method needs to take in a dataframe, and two column identifiers, and then calculates expectations for
        # each value combination of those columns (i.e. State=PA, Cell Phone State = CA) according to the internal logic of the
        # fitter
        pass

    def amount_above_expectations(self, empirical_counts):
        # This method takes in an empirical counts object (which is stored as a frequency matrix/dataframe)
        # and returns how far above expectations it is
        pass

class IndependenceFitFactorExpectations(ExpectationsObject):
    # This fitter assumes that our best expectation of the frequency within a
    # bin of X=x & Y=y, is equivalent to P(Y=y)*P(X=x). In other words,
    # we assume the feature values are independent across features. This has the
    # benefit of giving us more data to train on, and thus having more stable estimates,
    # but it has the disadvantage of being obviously a wrong assumption, since these
    # features are correlated to each other
    # This was implemented because it is the default density metric used in the CLICKS paper

    def __init__(self, expectation_floor=2):
        # we put in an expectations floor that determines at what threshold
        # we determine a point to be above expectations. By default this is set to 2 (i.e. a point is
        # "dense" if it has counts 2x higher than our expectation)
        self.expectations_floor = expectation_floor

    def fit(self, df, left_col, right_col):
        left_probabilities = df[left_col].value_counts(normalize=True).sort_index()
        right_probabilities = df[right_col].value_counts(normalize=True).sort_index()
        col_expected_proba = pd.DataFrame(np.outer(left_probabilities.values, right_probabilities.values),
                                          index=left_probabilities.index, columns=right_probabilities.index)
        ret = deepcopy(self)
        ret.expected_proba = col_expected_proba
        return ret

    def amount_above_expectations(self, empirical_counts):
        expected_counts = self.expected_proba * np.nansum(empirical_counts.values)
        expected_counts[expected_counts < self.expectations_floor] = self.expectations_floor
        return (empirical_counts / expected_counts)


class DependenceFitFactorExpectations(ExpectationsObject):
    # This is similar to the above Independence fitter, except that we estimate
    # P(X=x, Y=y) based on estimating each cell within the grid, rather than only
    # estimating the marginal probabilities

    def __init__(self, expectation_floor=2):
        self.expectations_floor = expectation_floor
        self.name = "DependenceFitFactorExpectations_{}".format(expectation_floor)

    def fit(self, df, left_col, right_col):
        neither_null = df[(pd.notnull(df[left_col])) & (pd.notnull(df[right_col]))]
        col_expected_proba = ((neither_null.groupby([left_col, right_col]).size() / len(neither_null))
                              .reset_index().pivot(left_col, right_col, values=0))
        ret = deepcopy(self)
        ret.expected_proba = col_expected_proba
        return ret

    def amount_above_expectations(self, empirical_counts):
        expected_counts = self.expected_proba * np.nansum(empirical_counts.values)
        expected_counts[expected_counts < self.expectations_floor] = self.expectations_floor
        return (empirical_counts / expected_counts)



class VarianceFit(ExpectationsObject):
    # This fitter incorporates historical variance, and uses that to scale the deviation
    # from historical mean value. This binner has historical been better at catching large rings,
    # because the DFF 2x multiplier on the absolute value of counts is too conservative when it comes to large rings
    # however, it is overly sensitive when it comes to small rings
    def __init__(self, time_var):
        self.time_var = time_var
        self.name = "VarianceFit_{}".format(time_var)

    def fit(self, df, left_col, right_col):
        transform_map = {'user_id': {'num_apps': lambda x: np.sum(pd.notnull(x))}}
        temporal_fit = (dp.DimensionalPipeline(df, 'variance_fit', False)
                        #Aggregate data along two categories and time variable
                        .aggregate_data([left_col, right_col, self.time_var], transform_map, debug=False)
                        #Fill gaps along time variable, such that each time slice is filled for all category intersections
                        .fill_gaps([left_col, right_col], self.time_var, {'num_apps': 0}, debug=False)
                        #Append back to time/category aggregated data a weekly count: think window function
                        #This quasi-windowing capability is brought in by broadcast_aggregate
                        .aggregate_data([self.time_var], {'num_apps': {'weekly_count': np.sum}},
                                        broadcast_aggregate=True, debug=False)
                        #Divide num_apps by weekly_count to get a percent for each cell within each time slice
                        .generic_apply(lambda x: np.divide(x['num_apps'], x['weekly_count']), 'percent_of_weekly_apps',
                                       debug=False)
                        #Aggregate down to get the mean (i.e. average cell multinomial proba across weeks) and sd
                        # i.e. root variance in multinomial proba from week to week
                        .aggregate_data([left_col, right_col], {'percent_of_weekly_apps': {'percent_mean': np.mean,
                                                                                      'percent_sd': lambda x: np.std(x, ddof=1)}}, debug=False)
                        )
        other_self = deepcopy(self)
        other_self.mean = temporal_fit.df.pivot(left_col, right_col, 'percent_mean')
        other_self.sd = temporal_fit.df.pivot(left_col, right_col, 'percent_sd')
        return other_self


    def amount_above_expectations(self, empirical_counts):
        total_count = np.nansum(empirical_counts)
        total_percentages = empirical_counts/float(total_count)

        return (total_percentages - self.mean)/self.sd

class WeightedVarianceFit(VarianceFit):
    # The point of the WeightedVarianceFit is to fix the issue described above that
    # the threshold should scale with size. This is sensible because we *care* much
    # more about larger anomalies, so we have a higher threshold for small anomalies to meet in order to
    # become raised to our attention
    def __init__(self, time_var, threshold_expansion=7):
        self.time_var = time_var
        self.name = "WeightedVarianceFit_{}".format(time_var)
        self.threshold_expansion = threshold_expansion

    def amount_above_expectations(self, empirical_counts):
        total_count = np.nansum(empirical_counts)
        total_percentages = empirical_counts/float(total_count)
        #Currently, this starts at 0 and then increases up to 7
        weighted_threshold = -self.threshold_expansion + 2*self.threshold_expansion*expit(total_percentages*10)
        return (total_percentages - self.mean)/self.sd - weighted_threshold

class MADFit(ExpectationsObject):
    # This fitter is like the variance fit, but instead of calculating difference from mean and scaling by
    # standard deviation, it calculates difference from median and scales by mean absolute deviation
    def __init__(self, time_var):
        self.time_var = time_var
        self.name = "MADFit_{}".format(time_var)

    def fit(self, df, left_col, right_col):
        #the transform map specifies: 1) in it's key, which column is being used as the base for the calculation
        # and 2) the names and callables of the calculations that will be built on that
        transform_map = {'user_id': {'num_apps': lambda x: np.sum(pd.notnull(x))}}
        temporal_fit = (dp.DimensionalPipeline(df, 'variance_fit', False)
                        #Aggregate data along two categories and time variable
                        .aggregate_data([left_col, right_col, self.time_var], transform_map, debug=False)
                        #Fill gaps along time variable, such that each time slice is filled for all category intersections
                        .fill_gaps([left_col, right_col], self.time_var, {'num_apps': 0}, debug=False)
                        #Append back to time/category aggregated data a weekly count: think window function
                        #This quasi-windowing capability is brought in by broadcast_aggregate
                        .aggregate_data([self.time_var], {'num_apps': {'weekly_count': np.sum}},
                                        broadcast_aggregate=True, debug=False)
                        #Divide num_apps by weekly_count to get a percent for each cell within each time slice
                        .generic_apply(lambda x: np.divide(x['num_apps'], x['weekly_count']), 'percent_of_weekly_apps',
                                       debug=False)
                        #Aggregate down to get the mean (i.e. average cell multinomial proba across weeks) and sd
                        # i.e. root variance in multinomial proba from week to week
                        .aggregate_data([left_col, right_col], {'percent_of_weekly_apps': {'percent_median': np.median,
                                                                                      'percent_mad': lambda x: mad(x)}}, debug=False)
                        )
        other_self = deepcopy(self)
        other_self.median = temporal_fit.df.pivot(left_col, right_col, 'percent_median')
        other_self.mad = temporal_fit.df.pivot(left_col, right_col, 'percent_mad')
        return other_self


    def amount_above_expectations(self, empirical_counts):
        total_count = np.nansum(empirical_counts)
        total_percentages = empirical_counts/float(total_count)

        return (total_percentages - self.median)/self.mad


class HistoricalFeatureExpectations():
    # A utility class that takes pairs of columns and fits expectations objects
    # for each pair of columns
    def __init__(self, relevant_dimensions, train_df, expectations_object):
        pair_array = list(combinations(relevant_dimensions, 2))
        expectations_map = {}
        ##Store expectations for each pair in an identically-indexed array (not a dict due to sets not being hashable)
        for column_set in pair_array:
            left_col, right_col = sorted(column_set)
            # Fit the expectations_fitter to each column
            # At some later point, we can have a different expectations object for each column pair,

            # but this is not yet implemented
            col_expectations_object = expectations_object.fit(train_df, left_col, right_col)
            expectations_map[left_col + '_' + right_col] = col_expectations_object

        self.expectations_map = expectations_map


    def get_expectations(self, col1, col2):
        left_col, right_col = sorted([col1, col2])
        return self.expectations_map[left_col + '_' + right_col]
