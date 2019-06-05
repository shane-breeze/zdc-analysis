#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import oyaml as yaml
import tqdm
import copy

import zdc.modules.dataframe as df_tools
import zdc.modules.shapes as shape_tools
import zdc.modules.datacard as dc_tools

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
    df_data = df_tools.preprocess(df_data, cfg["data"]["preproc"])
    df_data = df_tools.create_channel_processes(
        df_data, cfg["data"]["channel_processes"], [b[0] for b in binning],
    )
    df_data = df_tools.rebin(df_data, binning)
    df_data = df_tools.densify(df_data, binning)

    # mc
    print("Pre-processing MC")
    df_mc_jecs = df_tools.process_objsyst(df_mc_jecs)
    df_mc_leps = df_tools.process_objsyst(df_mc_leps)
    df_mc = pd.concat([df_mc, df_mc_jecs, df_mc_leps], axis=1).fillna(0.)
    df_mc = df_tools.preprocess(df_mc, cfg["mc"]["preproc"])
    df_mc = df_tools.create_channel_processes(
        df_mc, cfg["mc"]["channel_processes"], [b[0] for b in binning],
    )
    df_mc = df_tools.rebin(df_mc, binning)
    df_mc = df_tools.densify(df_mc, binning)

    # Down had a bug. Symmetrise for now
    df_mc.loc[:,"sum_w_jesTotalDown"] = 2*df_mc["sum_w"] - df_mc["sum_w_jesTotalUp"]
    df_mc.loc[:,"sum_ww_jesTotalDown"] = df_mc["sum_ww_jesTotalUp"]*(df_mc["sum_w_jesTotalDown"]/df_mc["sum_w_jesTotalUp"])**2
    df_mc.loc[:,"sum_w_jerSFDown"] = 2*df_mc["sum_w"] - df_mc["sum_w_jerSFUp"]
    df_mc.loc[:,"sum_ww_jerSFDown"] = df_mc["sum_ww_jerSFUp"]*(df_mc["sum_w_jerSFDown"]/df_mc["sum_w_jerSFUp"])**2

    # process additional systematics
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

    # Process PDF variations, alphaS and QCD scale
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
                up_smooth = shape_tools.smooth(xval, up_ratio, err=up_err)
                do_smooth = shape_tools.smooth(xval, do_ratio, err=do_err)

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

    df_shapes = shape_tools.create_shapes(tdf_data, tdf_mc, shape_var, systs)
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

    dc_tools.create_datacard(
        df_data, df_mc, systs, cfg["params"], dcpath, shapepath,
        "$CHANNEL:$PROCESS:central:$MASS,sum_w:sum_ww",
        "$CHANNEL:$PROCESS:$SYSTEMATIC:$MASS,sum_w:sum_ww",
    )

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
        draw_syst_templates(df_mc, cfg["mc"]["systs"], bins)

if __name__ == "__main__":
    main()
