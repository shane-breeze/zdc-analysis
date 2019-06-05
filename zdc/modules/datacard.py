"""Helper functions to create the datacards"""

import os
import tabulate as tab

def create_datacard(
    df_data, df_mc, systs, params, path, shapes, central_path, syst_path
):
    df_data = df_data.reset_index()
    df_mc = df_mc.reset_index()

    # heading
    dc = tab.tabulate([
        ["imax", "*", "number of channels"],
        ["jmax", "*", "number of channels"],
        ["kmax", "*", "number of channels"],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # shapes
    dc += tab.tabulate([
        ["shapes", "*", "*", os.path.basename(shapes), central_path, syst_path],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # data_obs
    obs = [list(x) for x in df_data[["channel", "sum_w"]].values.T]
    dc += tab.tabulate([
        ["bin"]+obs[0],
        ["observation"]+obs[1],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # expected
    rate = [list(x) for x in df_mc[["channel", "process", "procidx", "sum_w"]].values.T]
    dc += tab.tabulate([
        ["bin"] + rate[0],
        ["process"] + rate[1],
        ["process"] + rate[2],
        ["rate"] + rate[3],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # systematics
    df_mc = df_mc.groupby(["channel", "process", "procidx"]).sum()
    df_mc = df_mc[[c for c in df_mc.columns if c.startswith("sum_w_")]]
    df_mc.columns = [c.replace("sum_w_", "") for c in df_mc.columns]

    df_mc_up = df_mc[[c for c in df_mc.columns if c.endswith("Up")]]
    df_mc_up.columns = [c[:-2] for c in df_mc_up.columns]
    df_mc_up = df_mc_up[[c for c in df_mc_up.columns if c in systs]]
    df_mc_up.loc[:,:] = 1.

    df_mc_do = df_mc[[c for c in df_mc.columns if c.endswith("Down")]]
    df_mc_do.columns = [c[:-4] for c in df_mc_do.columns]
    df_mc_do = df_mc_do[[c for c in df_mc_do.columns if c in systs]]
    df_mc_do.loc[:,:] = 1.

    dc += tab.tabulate([
        [c, 'shape']+list(v)
        for c, v in zip(df_mc_up.columns.values.T.astype(str), df_mc_up.values.T)
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # params
    dc += tab.tabulate(params, [], tablefmt="plain")

    with open(path, 'w') as f:
        f.write(dc)
    print("Created {}".format(path))
