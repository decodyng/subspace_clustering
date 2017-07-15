import os
import numpy as np
import pandas as pd


## The goal of this calculation is to create an object that can do a few generic types of calculations necessary for aggregation along both 
## feature dimensions and time, and track its changes. This is also designed to make it easy to re-load and re-do calculations from fixed points 

#A better name would really be "TimeSeriesPipeline"

def deviation_calc(sr, obs_col='routing_percentage', mean_col='rolling_mean_twenty_five', stdev_col='rolling_stdev_twenty_five'):
    obs_perc = sr[obs_col]
    mn = sr[mean_col]
    sd = sr[stdev_col]
    if obs_perc == 0:
        return 0
    if pd.isnull(mn) or mn == 0 or pd.isnull(sd) or sd == 0:
        return np.nan
    else:
        return (obs_perc - mn)/sd


def idiotic_week_ts_conversion(ts_tup):
    st_tup = str(ts_tup) + '1'
    return pd.to_datetime(st_tup, format='(%Y, %U)%w')


class DimensionalPipeline(object):
    #We initialize a DimensionalPipeline object with an initial dataframe, a name (that also doubles as a path), and a boolean for should_save
    def __init__(self, df, name, should_save):
        self.df = df
        self.name = name
        self.should_save = should_save
        if self.should_save:
            if not os.path.exists(self.name):
                os.mkdir(self.name)
                df.to_csv(os.path.join(self.name, '_init.csv'), index=False)


    def aggregate_data(self, dimensions, transformations, step_name=None, debug=True, broadcast_aggregate=False):
        # This transformation is in many ways a typical group by, but it has two useful additional functions. 
        # One, it allows you to pass in arbitrary lambda transformations to your calculations. 
        # Two, it has an option for broadcast_aggregate, which makes the group by act more like a 
        # postgres window function, and broadcast the grouped calculation back as a column onto the ungrouped dataframe 

        if step_name is None:
            step_name = "agg_" + "".join(dimensions)
        
        if debug:
            print "On step " + step_name

        if os.path.exists(step_save_name):
            print "Version of this step already found, loading from disk"
            loaded = pd.read_csv(step_save_name)
            self.df = loaded
            return self

        gb = self.df.groupby(dimensions)
        i = 0
        for col, transforms in transformations.iteritems():
            if debug:
                print "Executing transforms on column " + col
            col_agg = gb[col].agg(transforms)
            if i == 0:
                merged = col_agg
            else:
                merged = pd.merge(merged, col_agg, how='left', left_index=True, right_index=True)
            i += 1

        #reset index to be meaningful 
        merged = merged.reset_index()
        if broadcast_aggregate:
            merged = pd.merge(self.df, merged, how='left', on=dimensions)
        self.df = merged

        if self.should_save:
            step_save_name = os.path.join(self.name, step_name + ".csv")
            self.df.to_csv(step_save_name, index=False)


        #For this transformation, like all of them, the self object is returned, so that operations can be chained
        return self

    def fill_gaps(self, dimensions, fill_index, fill_values, step_name=None, debug=True):
        # The goal of this step is to account for the fact that there may not natively be observations for every feature combination for every 
        # point in time, but we may still want to "normalize" across time, such that we can count nulls as 0 applications for that week, rather 
        # than simply skipping to the next week 

        if step_name is None:
            step_name = "fill_" + fill_index
        step_save_name = os.path.join(self.name, step_name + ".csv")

        if debug:
            print "On step " + step_name

        if os.path.exists(step_save_name):
            print "Version of this step already found, loading from disk"
            loaded = pd.read_csv(step_save_name)
            self.df = loaded
            return self

        all_ind_vals = self.df[fill_index].unique()
        all_filled_dfs = []

        for dim_vals, grouped in self.df.groupby(dimensions):
            grouped = grouped.set_index(fill_index)
            grouped.index.rename(fill_index, inplace=True)
            reindexed = grouped.reindex(all_ind_vals)
            reindexed = reindexed.reset_index()
            reindexed[dimensions] = dim_vals
            all_filled_dfs.append(reindexed)

        filled_df = pd.concat(all_filled_dfs)
        for k, v in fill_values.iteritems():
            filled_df[k] = filled_df[k].fillna(v)
        self.df = filled_df

        if self.should_save:
            self.df.to_csv(step_save_name, index=False)
        return self

    def rolling_calculation(self, group_dim, sort_col, agg_callables, step_name=None, debug=True, keep_cols=None):
        #This transformation conducts a rolling calculation on the dataframe, and then afterwards, keep access to columns 
        # from the original dataframe 
        
        if step_name is None:
            step_name = "rolling_on_" + sort_col
        step_save_name = os.path.join(self.name, step_name + ".csv")

        if debug:
            print "On step " + step_name

        if os.path.exists(step_save_name):
            print "Version of this step already found, loading from disk"
            loaded = pd.read_csv(step_save_name)
            self.df = loaded
            return self

        if keep_cols is None:
            keep_cols = set(self.df.columns).difference(set([group_dim, sort_col]))

        rolling_dfs = []
        working_df = self.df.sort_values(by=sort_col).set_index(sort_col)
        print self.df.columns

        for group_val, grouped in working_df.groupby(group_dim):
            for i, k in enumerate(agg_callables.keys()):
                call_info = agg_callables[k]
                rolling_val = grouped[call_info["col"]] \
                    .rolling(window=call_info["window"], min_periods=call_info["min_periods"]) \
                    .apply(call_info["call"]) \
                    .reset_index()
                rolling_val.columns = [sort_col, k]

                if i == 0:
                    merged = rolling_val
                else:
                    merged = pd.merge(merged, rolling_val, how='left', left_on=sort_col, right_on=sort_col)

            for col in keep_cols:
                merged[col] = grouped[col].values
            merged[group_dim] = group_val
            rolling_dfs.append(merged)

        rolling_concat = pd.concat(rolling_dfs)
        self.df = rolling_concat
        if self.should_save:
            self.df.to_csv(step_save_name, index=False)

        return self

    def generic_apply(self, callable_func, output_name, step_name=None, col=None, debug=True):
        #This allows you to use a generic apply, either on a full row, or on column within a row
        if step_name is None:
            step_name = "apply_to_generate_" + output_name
        step_save_name = os.path.join(self.name, step_name + ".csv")

        if debug:
            print "On step " + step_name

        if os.path.exists(step_save_name):
            print "Version of this step already found, loading from disk"
            loaded = pd.read_csv(step_save_name)
            self.df = loaded

        if col is None:
            # if col is none, assumes acting on entire DF/row object
            self.df[output_name] = self.df.apply(callable_func, axis=1)
        else:
            self.df[output_name] = self.df[col].apply(callable_func)
        if self.should_save:
            self.df.to_csv(step_save_name, index=False)

        return self