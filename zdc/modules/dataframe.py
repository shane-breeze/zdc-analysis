"""Helper functions to manipulate dataframe"""

import numpy as np
import pandas as pd

def preprocess(df, evals):
    for ev in evals:
        f = eval("lambda "+ev)
        try:
            ret = f(df)
        except AttributeError as err:
            raise AttributeError(err, ev)
        if ret is not  None:
            df = ret
    return df

def create_channel_processes(df, channel_processes, binvars):
    dfs = []

    df["channel"] = ""
    df["process"] = ""
    df["procidx"] = np.nan

    idx_names = ["channel", "process", "procidx"]+binvars
    col_names = [c for c in df.columns if c.startswith("sum_w")]

    for chan_proc in channel_processes:
        df_select = eval("lambda "+chan_proc["eval"])(df).copy(deep=True)
        df_select.loc[:, "channel"] = chan_proc["name"][0]
        df_select.loc[:, "process"] = chan_proc["name"][1]
        df_select.loc[:, "procidx"] = chan_proc["name"][2]
        dfs.append(df_select[idx_names+col_names])

    new_df = pd.concat(dfs)
    new_df.loc[:, "procidx"] = new_df["procidx"].astype(int)
    return new_df.set_index(idx_names)

def rebin(df, binning):
    for label, bins in binning:
        idx = df.index.to_frame().reset_index(drop=True)
        idx.loc[:, label] = bins[np.minimum(bins.searchsorted(idx[label]), len(bins)-1)]
        df.index = pd.MultiIndex.from_frame(idx)
    return df

def densify(df, binning):
    binvars = [b[0] for b in binning]

    idx_names = ["channel", "process", "procidx"]+binvars
    col_names = [c for c in df.columns if c.startswith("sum_w")]
    df = df.groupby(idx_names).sum()[col_names]

    data = {n: [] for n in idx_names}
    nbins = np.prod([b[1].shape[0] for b in binning])
    for (ch, proc, pid), _ in df.groupby(["channel", "process", "procidx"]):
        data["channel"].extend([ch]*nbins)
        data["process"].extend([proc]*nbins)
        data["procidx"].extend([pid]*nbins)

        for idx, (label, bins) in enumerate(binning):
            nbins_above = np.prod([1]+[b[1].shape[0] for b in binning[:idx]])
            nbins_below = np.prod([1]+[b[1].shape[0] for b in binning[idx+1:]])

            curr_bins = []
            for b in bins:
                curr_bins.extend([b]*nbins_below)
            curr_bins = curr_bins*nbins_above
            data[label].extend(curr_bins)

    idx = pd.MultiIndex.from_frame(pd.DataFrame(data, columns=idx_names))
    return df.reindex(idx).fillna(0.)

def process_objsyst(df):
    df = df.pivot_table(
        values=["sum_w", "sum_ww"], columns=["table"],
        index=[n for n in df.index.names if n!="table"],
        aggfunc=np.sum, fill_value=0.,
    )
    cols = df.columns.levels
    new_cols = ["_".join([c0, c1]) for c0 in cols[0] for c1 in cols[1]]
    df.columns = new_cols
    return df
