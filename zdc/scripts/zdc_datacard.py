#!/usr/bin/env python
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import oyaml as yaml
import itertools as it
import root_numpy
import tabulate as tab
import tqdm
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", type=str, help="Input path")
    parser.add_argument("cfg", type=str, help="Input config yaml")
    parser.add_argument("-d", "--datacard", type=str, default="datacard.txt",
                        help="Output path for the datacard")
    parser.add_argument("-s", "--shape", type=str, default="shape.h5",
                        help="Output path for the shape file")
    parser.add_argument("--draw", default=False, action='store_true',
                        help="Draw alternative templates")
    return parser.parse_args()

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

def open_dfs(path, cfg, systs):
    with pd.HDFStore(path) as store:
        print("Reading data")
        df_data = store["DataAggEvents"].loc[("central",), :]
        print("Reading MC")
        df_mc = store["MCAggEvents"].loc[("central",), :]
        print("Reading MC (jecs)")
        df_mc_jecs = store["MCAggEvents_jecs"]
        print("Reading MC (leps)")
        df_mc_leps = store["MCAggEvents_leps"]

    binning = [(sbins[0], eval(sbins[1])) for sbins in cfg["binning"]]

    # data
    print("Pre-processing data")
    df_data = preprocess(df_data, cfg["data"]["preproc"])
    df_data = create_channel_processes(
        df_data, cfg["data"]["channel_processes"], [b[0] for b in binning],
    )
    df_data = rebin(df_data, binning)
    df_data = densify(df_data, binning)

    # mc
    print("Pre-processing MC")
    df_mc_jecs = process_objsyst(df_mc_jecs)
    df_mc_leps = process_objsyst(df_mc_leps)
    df_mc = pd.concat([df_mc, df_mc_jecs, df_mc_leps], axis=1).fillna(0.)
    df_mc = preprocess(df_mc, cfg["mc"]["preproc"])
    df_mc = create_channel_processes(
        df_mc, cfg["mc"]["channel_processes"], [b[0] for b in binning],
    )
    df_mc = rebin(df_mc, binning)
    df_mc = densify(df_mc, binning)

    df_mc.loc[:,"sum_w_jesTotalDown"] = 2*df_mc["sum_w"] - df_mc["sum_w_jesTotalUp"]
    df_mc.loc[:,"sum_ww_jesTotalDown"] = df_mc["sum_ww_jesTotalUp"]*(df_mc["sum_w_jesTotalDown"]/df_mc["sum_w_jesTotalUp"])**2
    df_mc.loc[:,"sum_w_jerSFDown"] = 2*df_mc["sum_w"] - df_mc["sum_w_jerSFUp"]
    df_mc.loc[:,"sum_ww_jerSFDown"] = df_mc["sum_ww_jerSFUp"]*(df_mc["sum_w_jerSFDown"]/df_mc["sum_w_jerSFUp"])**2

    df_add_systs = pd.DataFrame()
    for name, vals in cfg["mc"]["add_syst"].items():
        vals = np.array(vals)

        tdf_add = pd.DataFrame()
        for cat, dfg in df_mc.groupby(["channel", "process", "procidx"]):
            tdf_up = dfg["sum_w"]*(1+vals)
            tdf_do = dfg["sum_w"]*(1-vals)
            tdf_up.name = "sum_w_{}Up".format(name)
            tdf_do.name = "sum_w_{}Down".format(name)

            tdf_wwup = dfg["sum_ww"]*(1+vals)**2
            tdf_wwdo = dfg["sum_ww"]*(1-vals)**2
            tdf_wwup.name = "sum_ww_{}Up".format(name)
            tdf_wwdo.name = "sum_ww_{}Down".format(name)

            tdf = pd.concat([tdf_up, tdf_do, tdf_wwup, tdf_wwdo], axis=1)
            tdf_add = pd.concat([tdf_add, tdf], axis=0)
        df_add_systs = pd.concat([df_add_systs, tdf_add], axis=1)
    df_mc = pd.concat([df_mc, df_add_systs], axis=1)

    pdf_cols = [c for c in df_mc.columns if "LHEPdfWeight" in c]
    pdf_std = df_mc.loc[:, pdf_cols].std(axis=1)
    df_mc["sum_w_pdfscaleUp"] = df_mc["sum_w"] + pdf_std
    df_mc["sum_ww_pdfscaleUp"] = df_mc.eval("sum_ww*(sum_w_pdfscaleUp/sum_w)**2")
    df_mc["sum_w_pdfscaleDown"] = df_mc["sum_w"] - pdf_std
    df_mc["sum_ww_pdfscaleDown"] = df_mc.eval("sum_ww*(sum_w_pdfscaleDown/sum_w)**2")
    df_mc = df_mc.drop(pdf_cols, axis=1)

    df_mc["sum_ww_alphasUp"] = df_mc.eval("(sum_w_alphasUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_alphasDown"] = df_mc.eval("(sum_w_alphasDown/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFUp"] = df_mc.eval("(sum_w_qcdFUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFDown"] = df_mc.eval("(sum_w_qcdFDown/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdRUp"] = df_mc.eval("(sum_w_qcdRUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdRDown"] = df_mc.eval("(sum_w_qcdRDown/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFqcdRUp"] = df_mc.eval("(sum_w_qcdFqcdRUp/sum_w)**2 * sum_ww")
    df_mc["sum_ww_qcdFqcdRDown"] = df_mc.eval("(sum_w_qcdFqcdRDown/sum_w)**2 * sum_ww")

    df_data = df_data.fillna(0.)
    df_mc = df_mc.fillna(0.)

    binvars = [b for b in df_mc.index.names if "binvar" in b]
    df_smooths = []
    print("Smoothing MC alternative templates")
    for cat, dfg in tqdm.tqdm(
            df_mc.groupby(["channel", "process", "procidx"]+binvars[:-1]),
    ):
        df_cat = pd.DataFrame({}, columns=[], index=dfg.index)
        binlow = dfg.index.get_level_values(binvars[-1]).values
        binedge = np.array(list(binlow)+[2*binlow[-1]-binlow[-2]])
        xval = (binedge[1:] + binedge[:-1])/2.

        df_ratio = dfg.divide(dfg["sum_w"], axis=0)
        for syst in systs:
            up_ratio = df_ratio["sum_w_{}Up".format(syst)].values
            do_ratio = df_ratio["sum_w_{}Down".format(syst)].values

            if (np.all(up_ratio==1.) and np.all(do_ratio==1.)) or (np.all(up_ratio==0.) and np.all(do_ratio==0.)):
                up_smooth, up_scale = 1., 1.
                do_smooth, do_scale = 1., 1.
            else:
                up_err = dfg.eval("sqrt(abs(sum_ww_{0}Up - sum_ww_{0}Down))/sum_w".format(syst))
                do_err = dfg.eval("sqrt(abs(sum_ww_{0}Down - sum_ww))/sum_w".format(syst))
                up_err[up_ratio==0.] = 0.01
                do_err[do_ratio==0.] = 0.01

                # smoothed
                up_smooth = smooth(xval, up_ratio, err=up_err)
                do_smooth = smooth(xval, do_ratio, err=do_err)

                up_denom = (up_smooth*dfg["sum_w"]).sum()
                do_denom = (do_smooth*dfg["sum_w"]).sum()
                if dfg["sum_w"].sum() != 0. and up_denom != 0. and do_denom != 0.:
                    up_scale = dfg["sum_w_{}Up".format(syst)].sum() / up_denom
                    do_scale = dfg["sum_w_{}Down".format(syst)].sum() / do_denom
                else:
                    up_scale = 1.
                    do_scale = 1.

            df_cat.loc[:,"sum_w_{}_smoothUp".format(syst)] = dfg["sum_w"]*up_smooth*up_scale
            df_cat.loc[:,"sum_w_{}_smoothDown".format(syst)] = dfg["sum_w"]*do_smooth*do_scale
            df_cat.loc[:,"sum_ww_{}_smoothUp".format(syst)] = dfg["sum_ww_{}Up".format(syst)]
            df_cat.loc[:,"sum_ww_{}_smoothDown".format(syst)] = dfg["sum_ww_{}Down".format(syst)]

            # flat
            if dfg["sum_w"].sum() != 0.:
                df_cat.loc[:,"sum_w_{}_flatUp".format(syst)] = dfg["sum_w"]*dfg["sum_w_{}Up".format(syst)].sum()/dfg["sum_w"].sum()
                df_cat.loc[:,"sum_w_{}_flatDown".format(syst)] = dfg["sum_w"]*dfg["sum_w_{}Down".format(syst)].sum()/dfg["sum_w"].sum()
                df_cat.loc[:,"sum_ww_{}_flatUp".format(syst)] = dfg["sum_ww"]*(dfg["sum_w_{}Up".format(syst)].sum()/dfg["sum_w"].sum())**2
                df_cat.loc[:,"sum_ww_{}_flatDown".format(syst)] = dfg["sum_ww"]*(dfg["sum_w_{}Down".format(syst)].sum()/dfg["sum_w"].sum())**2
            else:
                df_cat.loc[:,"sum_w_{}_flatUp".format(syst)] = dfg["sum_w"]
                df_cat.loc[:,"sum_w_{}_flatDown".format(syst)] = dfg["sum_w"]
                df_cat.loc[:,"sum_ww_{}_flatUp".format(syst)] = dfg["sum_ww"]
                df_cat.loc[:,"sum_ww_{}_flatDown".format(syst)] = dfg["sum_ww"]

        df_smooths.append(df_cat)
    df_smooths = pd.concat(df_smooths, axis=0, sort=False).fillna(0.)
    df_mc = pd.concat([df_mc, df_smooths], axis=1, sort=False)

    return df_data, df_mc

def smooth(x, y, err=None):
    if err is not None:
        s = UnivariateSpline(x, y, w=1./err, k=2, s=x.shape[0]*4)
    else:
        s = UnivariateSpline(x, y, k=2, s=x.shape[0]*4)
    return gaussian_filter1d(s(x), 1)

def create_shapes(idf_data, idf_mc, shape_var, systematics):
    df_data = idf_data.reset_index("procidx", drop=True)
    df_data.loc[:,"systematic"] = "central"
    index = list(df_data.index.names)
    index.insert(index.index("process")+1, "systematic")
    df_data = df_data.reset_index().set_index(index).sort_index()

    df_mc = idf_mc.reset_index("procidx", drop=True)
    df_mc = df_mc.stack().to_frame("value")
    df_mc.index.names = list(df_mc.index.names)[:-1] + ["systematic"]
    df_mc = df_mc.reset_index("systematic")
    df_mc.loc[:,"sum"] = np.where(df_mc["systematic"].str.startswith("sum_ww"), "sum_ww", "sum_w")
    df_mc.loc[:,"systematic"] = (
        df_mc["systematic"]
        .str.replace("sum_w_","")
        .str.replace("sum_ww_", "")
        .str.replace("sum_ww", "central")
        .str.replace("sum_w", "central")
    )
    df_mc = pd.pivot_table(
        df_mc, values=["value"], columns=["sum"],
        index=list(df_mc.index.names)+["systematic"], aggfunc=np.sum,
    )
    df_mc.columns = ["sum_w", "sum_ww"]
    df_mc = df_mc.reset_index().set_index(index).sort_index()

    df_shapes = pd.concat([df_data, df_mc], axis=0, sort=True)
    return df_shapes

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

def create_fitinputs(tdf_data, tdf_mc, shapepath, dcpath, cfg, systs):
    if not cfg["overflow"]:
        index = tdf_data.index.names
        tdf_data = (
            tdf_data
            .groupby(list(index)[:-1], as_index=False)
            .apply(lambda x: x.iloc[:-1])
        ).reset_index().set_index(index).sort_index()[["sum_w", "sum_ww"]]

        index = tdf_mc.index.names
        tdf_mc = (
            tdf_mc
            .groupby(list(index)[:-1], as_index=False)
            .apply(lambda x: x.iloc[:-1])
        ).reset_index().set_index(index).sort_index()[["sum_w", "sum_ww"]]

    binvars = [sbins[0] for sbins in cfg["binning"]]
    binning = [(sbins[0], eval(sbins[1])) for sbins in cfg["binning"]]
    shape_var = binning.pop()[0]

    df_shapes = create_shapes(tdf_data, tdf_mc, shape_var, systs)
    df_shapes.to_hdf(
        shapepath, "shapes", format='table', append=False,
        complib='blosc:lz4hc', complevel=9,
    )

    df_data = tdf_data.groupby(
        [idx for idx in tdf_data.index.names if idx not in binvars]
    ).sum()
    df_data = df_data.astype(int)
    df_data.loc[:,:] = -1

    df_mc = tdf_mc.groupby(
        [idx for idx in tdf_mc.index.names if idx not in binvars]
    ).sum()
    df_mc = df_mc.astype(int)
    df_mc.loc[:,:] = -1

    create_datacard(
        df_data, df_mc, systs, cfg["params"], dcpath, shapepath,
        "$CHANNEL:central:$PROCESS:$MASS,sum_w:sum_ww",
        "$CHANNEL:$SYSTEMATIC:$PROCESS:$MASS,sum_w:sum_ww",
    )

def draw_hist(x, w, **kwargs):
    plt.hist(x, weights=w, **kwargs)

def draw_fill(x, y1, y2, **kwargs):
    plt.fill_between(x, y1, y2, **kwargs)

def draw_syst_templates(df, systs, bins):
    binvars = [n for n in df.index.names if "binvar" in n]
    for cat, dfg in df.groupby(["channel"]+binvars[:-1]):
        cat = [cat] if len(binvars)==1 else cat
        df_syst = dfg[[c for c in dfg.columns if "sum_w_" in c and not "_smooth" in c and not "_flat" in c]]
        df_syst = df_syst.divide(dfg["sum_w"], axis=0) - 1.
        index = [n.capitalize() for n in df_syst.index.names]
        cols = [c.replace("sum_w_", "") for c in df_syst.columns]
        df_syst.columns = cols

        df_syst = df_syst.stack().reset_index()
        df_syst.columns = index + ["Syst", "nevents"]
        df_syst.loc[:,"Variation"] = np.where(df_syst["Syst"].str.endswith("Up"), 'Up', 'Down')
        df_syst.loc[:,"Syst"] = df_syst["Syst"].apply(lambda x: (x[:-2] if x.endswith("Up") else x[:-4]).replace("_", ""))

        df_smooth = dfg[[c for c in dfg.columns if "sum_w_" in c and "_smooth" in c]]
        df_smooth = df_smooth.divide(dfg["sum_w"], axis=0) - 1.
        df_smooth[df_smooth==-1.] = 0.
        index = [n.capitalize() for n in df_smooth.index.names]
        cols = [c.replace("sum_w_", "").replace("_smooth", "") for c in df_smooth.columns]
        df_smooth.columns = cols

        df_smooth = df_smooth.stack().reset_index()
        df_smooth.columns = index + ["Syst", "nevents smooth"]
        df_smooth.loc[:,"Variation"] = np.where(df_smooth["Syst"].str.endswith("Up"), 'Up', 'Down')
        df_smooth.loc[:,"Syst"] = df_smooth["Syst"].apply(lambda x: (x[:-2] if x.endswith("Up") else x[:-4]).replace("_", ""))

        df_flat = dfg[[c for c in dfg.columns if "sum_w_" in c and "_flat" in c]]
        df_flat = df_flat.divide(dfg["sum_w"], axis=0) - 1.
        index = [n.capitalize() for n in df_flat.index.names]
        cols = [c.replace("sum_w_", "").replace("_flat", "") for c in df_flat.columns]
        df_flat.columns = cols

        df_flat = df_flat.stack().reset_index()
        df_flat.columns = index + ["Syst", "nevents flat"]
        df_flat.loc[:,"Variation"] = np.where(df_flat["Syst"].str.endswith("Up"), 'Up', 'Down')
        df_flat.loc[:,"Syst"] = df_flat["Syst"].apply(lambda x: (x[:-2] if x.endswith("Up") else x[:-4]).replace("_", ""))

        index.extend(["Syst", "Variation"])
        df_syst = pd.concat([
            df_syst.set_index(index), df_smooth.set_index(index),
            df_flat.set_index(index),
        ], axis=1).reset_index()

        name = "alttemps_{}.pdf".format("_".join(map(str, cat)))
        df_syst.columns = [c if c!="Process" else "Proc" for c in df_syst.columns]
        g = sns.FacetGrid(
            df_syst, row='Syst', col='Proc', hue='Variation',
            row_order=[s for s in systs if s in df_syst["Syst"].unique()],
            hue_order=["Up", "Down"], sharex=True, sharey='row',
            margin_titles=True, xlim=(bins.min(), bins.max()),
            #ylim=(-0.075, 0.075),
        )
        g.map(
            draw_hist, binvars[-1].capitalize(), "nevents", bins=bins,
            histtype='step',
        )
        g.add_legend()
        g.map(
            draw_hist, binvars[-1].capitalize(), "nevents smooth", bins=bins,
            histtype='step', ls='--', color='black', label="Smooth",
        )
        g.map(
            draw_hist, binvars[-1].capitalize(), "nevents flat", bins=bins,
            histtype='step', ls='--', color='red', label="Flat",
        )
        g.set_axis_labels(r'$p_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)', "Relative template")

        g.fig.savefig(name, format='pdf', bbox_inches='tight')
        plt.close(g.fig)
        print("Created {}".format(name))

def main():
    options = parse_args()
    with open(options.cfg, 'r') as f:
        cfg = yaml.full_load(f)

    systs = copy.deepcopy(cfg["mc"]["systs"])
    df_data, df_mc = open_dfs(options.inpath, cfg, systs)

    systs += [s+"_smooth" for s in cfg["mc"]["systs"]]
    systs += [s+"_flat" for s in cfg["mc"]["systs"]]
    create_fitinputs(
        df_data, df_mc, options.shape, options.datacard, cfg, systs,
    )

    if options.draw:
        bins = eval(cfg["binning"][-1][1])
        bins = np.array(list(bins) + [2*bins[-1]-bins[-2]])
        plt.style.use('cms-sns')
        draw_syst_templates(df_mc, cfg["mc"]["systs"], bins)

if __name__ == "__main__":
    main()
