#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

import argparse
import numpy as np
import pandas as pd
import oyaml as yaml
import itertools as it
import root_numpy
import tabulate as tab

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", type=str, help="Input path")
    parser.add_argument("cfg", type=str, help="Input config yaml")
    parser.add_argument("-d", "--datacard", type=str, default="datacard.txt",
                        help="Output path for the datacard")
    parser.add_argument("-s", "--shape", type=str, default="shape.root",
                        help="Output path for the shape file")
    return parser.parse_args()

def change_parent(df, name, selection):
    df.loc[selection, "parent"] = name
    return df

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

def open_dfs(path, cfg):
    with pd.HDFStore(path) as store:
        df_data = store["DataAggEvents"].loc[("central",), :]
        df_mc = store["MCAggEvents"].loc[("central",), :]
        df_mc_jecs = store["MCAggEvents_jecs"]
        df_mc_leps = store["MCAggEvents_leps"]

    binning = [(sbins[0], eval(sbins[1])) for sbins in cfg["binning"]]

    # data
    df_data = preprocess(df_data, cfg["data"]["preproc"])
    df_data = create_channel_processes(
        df_data, cfg["data"]["channel_processes"], [b[0] for b in binning],
    )
    df_data = rebin(df_data, binning)
    df_data = densify(df_data, binning)

    # mc
    df_mc_jecs = process_objsyst(df_mc_jecs)
    df_mc_leps = process_objsyst(df_mc_leps)
    df_mc = pd.concat([df_mc, df_mc_jecs, df_mc_leps], axis=1).fillna(0.)
    df_mc = preprocess(df_mc, cfg["mc"]["preproc"])
    df_mc = create_channel_processes(
        df_mc, cfg["mc"]["channel_processes"], [b[0] for b in binning],
    )
    df_mc = rebin(df_mc, binning)
    df_mc = densify(df_mc, binning)
    df_mc["pdfscaleStdDev"] = df_mc.eval("sqrt(sum_ww_pdfscale - sum_w_pdfscale**2)/100.")
    df_mc["sum_w_pdfscaleUp"] = df_mc.eval("sum_w + pdfscaleStdDev")
    df_mc["sum_ww_pdfscaleUp"] = df_mc.eval("(sum_w_pdfscaleUp/sum_w)**2 * sum_ww")
    df_mc["sum_w_pdfscaleDown"] = df_mc.eval("sum_w - pdfscaleStdDev")
    df_mc["sum_ww_pdfscaleDown"] = df_mc.eval("(sum_w_pdfscaleDown/sum_w)**2 * sum_ww")

    df_mc["sum_ww_alphasUp"] = df_mc.eval("(sum_w_alphasUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_alphasDown"] = df_mc.eval("(sum_w_alphasDown/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFUp"] = df_mc.eval("(sum_w_qcdFUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFDown"] = df_mc.eval("(sum_w_qcdFDown/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdRUp"] = df_mc.eval("(sum_w_qcdRUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdRDown"] = df_mc.eval("(sum_w_qcdRDown/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFqcdRUp"] = df_mc.eval("(sum_w_qcdFqcdRUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFqcdRDown"] = df_mc.eval("(sum_w_qcdFqcdRDown/sum_w)**2 * sum_ww")

    return df_data, df_mc

def binned_df(df, idx, binning):
    idx_names = [i for i in idx if i not in [l for l, _ in binning]]
    for bins in it.product(*[b for _, b in binning]):
        selection = True
        for idx, (label, _) in enumerate(binning):
            selection = selection & (df[label]==bins[idx])
        tdf = df.drop([l for l, _ in binning], axis='columns')
        tdf = tdf.loc[selection].set_index(idx_names)
        yield tdf

def create_shapes(df_data, df_mc, shape_var, systematics, shape_file, direc):
    if direc not in [k.GetName() for k in shape_file.GetListOfKeys()]:
        shape_file.mkdir(direc)
    shape_file.cd(direc)
    for (ch, proc), dfg in df_mc.groupby(["channel", "process"]):
        binning = list(dfg.index.get_level_values(shape_var))
        last_bin = 2*binning[-1] - binning[-2]
        binning.append(last_bin)
        binning = np.array(binning).astype(float)

        # nominal
        rhist = ROOT.TH1D(proc, ", ".join([ch, proc]), binning.shape[0]-1, binning)
        rhist = root_numpy.array2hist(
            dfg["sum_w"].values, rhist, errors=np.sqrt(dfg["sum_ww"]).values,
        )

        if ch not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            ROOT.gDirectory.mkdir(ch)
        ROOT.gDirectory.cd(ch)
        if "central" not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            ROOT.gDirectory.mkdir("central")
        ROOT.gDirectory.cd("central")
        rhist.Write()
        rhist.Delete()
        shape_file.cd(direc)

        # systs
        for tsyst in systematics:
            for vari in ["Up", "Down"]:
                syst = tsyst + vari
                rhist = ROOT.TH1D(proc, ", ".join([ch, proc]), binning.shape[0]-1, binning)
                rhist = root_numpy.array2hist(
                    dfg["sum_w_{}".format(syst)].values, rhist,
                    errors=np.sqrt(dfg["sum_ww_{}".format(syst)]).values,
                )

                if ch not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
                    ROOT.gDirectory.mkdir(ch)
                ROOT.gDirectory.cd(ch)
                if syst not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
                    ROOT.gDirectory.mkdir(syst)
                ROOT.gDirectory.cd(syst)
                rhist.Write()
                rhist.Delete()
                shape_file.cd(direc)

    for (ch, proc), dfg in df_data.groupby(["channel", "process"]):
        binning = list(dfg.index.get_level_values(shape_var))
        last_bin = 2*binning[-1] - binning[-2]
        binning.append(last_bin)
        binning = np.array(binning).astype(float)

        # nominal
        rhist = ROOT.TH1D(proc, ", ".join([ch, proc]), binning.shape[0]-1, binning)
        rhist = root_numpy.array2hist(
            dfg["sum_w"].values, rhist, errors=np.sqrt(dfg["sum_ww"]).values,
        )

        if ch not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            ROOT.gDirectory.mkdir(ch)
        ROOT.gDirectory.cd(ch)
        if "central" not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            ROOT.gDirectory.mkdir("central")
        ROOT.gDirectory.cd("central")
        rhist.Write()
        rhist.Delete()
        shape_file.cd(direc)

    shape_file.cd()

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
        ["shapes", "*", "*", shapes, central_path, syst_path],
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

def create_fitinputs(df_data, df_mc, shapepath, dcpath, cfg):
    binning = [(sbins[0], eval(sbins[1])) for sbins in cfg["binning"]]
    rfile = ROOT.TFile.Open(shapepath, "RECREATE")
    shape_var = binning.pop()[0]
    bprod = it.product(*[b for _, b in binning])
    tdfs_data = binned_df(df_data.reset_index(), df_data.index.names, binning)
    tdfs_mc = binned_df(df_mc.reset_index(), df_mc.index.names, binning)

    for bins, tdf_data, tdf_mc in zip(bprod, tdfs_data, tdfs_mc):
        cat = "bin_"+"_".join(map(str, bins))
        if not cfg["overflow"]:
            tdf_data = tdf_data.iloc[:-1]
            tdf_mc = tdf_mc.iloc[:-1]
        create_shapes(
            tdf_data, tdf_mc, shape_var, cfg["mc"]["systs"], rfile, cat,
        )
        print("Created {}:{}".format(shapepath, cat))
        tdf_data = tdf_data.groupby(["channel", "process", "procidx"]).sum()
        tdf_mc = tdf_mc.groupby(["channel", "process", "procidx"]).sum()
        create_datacard(
            tdf_data, tdf_mc, cfg["mc"]["systs"], cfg["params"],
            dcpath.replace(".txt", "_"+cat+".txt"), shapepath,
            "bin_$MASS/$CHANNEL/central/$PROCESS",
            "bin_$MASS/$CHANNEL/$SYSTEMATIC/$PROCESS",
        )

    rfile.Close()
    rfile.Delete()

def main():
    options = parse_args()
    with open(options.cfg, 'r') as f:
        cfg = yaml.full_load(f)
    df_data, df_mc = open_dfs(options.inpath, cfg)
    create_fitinputs(df_data, df_mc, options.shape, options.datacard, cfg)

if __name__ == "__main__":
    main()
