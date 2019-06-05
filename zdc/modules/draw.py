import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_hist(x, w, **kwargs):
    plt.hist(x, weights=w, **kwargs)

def draw_fill(x, y1, y2, **kwargs):
    plt.fill_between(x, y1, y2, **kwargs)

def draw_syst_templates(df, systs, bins):
    plt.style.use('cms-sns')
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

