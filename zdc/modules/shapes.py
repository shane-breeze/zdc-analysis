"""Helper functions to create and manipulate shapes"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

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
