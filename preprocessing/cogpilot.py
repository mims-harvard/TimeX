from argparse import ArgumentParser
import math
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import signal


# datenum value of the Unix epoch start (19700101)
T0 = 719529

DAY_TO_SEC = 3600 * 24
MIN_TO_SEC = 60

# non-informitive EYE signals
EYE_VARS_RM = [
    "validity_l",
    "validity_r",
    "pupil_position_l_x",
    "pupil_position_r_x",
    "pupil_position_r_z",
    "convergence_distance_mm",
    "convergence_distance_validity",
]

# Subjects with missing signals
SUBJ_RM = [
    "sub-cp005",
    "sub-cp008",
    "sub-cp028",
    "sub-cp009",
    "sub-cp003",
    "sub-cp027",
]


def load_time_series(data_root, subject, sess_label, exp_type, modality, run_label):

    subject_run_data_dir = Path(data_root) / exp_type / subject / sess_label / run_label

    if modality.find("feat-perfmetric") != -1:
        modality_datafile = (
            f"{subject}_{sess_label}_{exp_type}_stream-{modality}_{run_label}.csv"
        )
    else:
        modality_datafile = f"{subject}_{sess_label}_{exp_type}_stream-{modality}_feat-chunk_{run_label}_dat.csv"

    datafile = subject_run_data_dir / modality_datafile
    dfEDA = pd.read_csv(datafile)

    return dfEDA


def resample_signal(x, t, fs_orig, FS, plots=False, nperiods=1, label=""):
    num_periods = len(x) // fs_orig
    (x_rs, t_rs) = signal.resample(x, num=FS * num_periods, t=t)
    return x_rs, t_rs


def resample_df(df, var_names, fs_orig, FS, t0, plots=False, nperiods=1, label=""):
    L_rs = []
    for i in range(len(var_names)):
        x = df[var_names[i]]
        t = df["time_dn"] - t0
        x_rs, t_rs = resample_signal(
            x, t, fs_orig, FS, plots=plots, nperiods=nperiods, label=label
        )
        L_rs.append((x_rs, t_rs))
    return L_rs


def segment_signal(x, numsegs, seglen, overlaplen, len_time):
    X = np.zeros((numsegs, seglen))
    start = 0
    seg = 0
    while start + seglen < len_time:
        X[seg, :] = x[start : start + seglen]
        start += overlaplen
        seg += 1
    return X


def remove_vars(df, var_names, var_names_rm):
    for i in range(len(var_names_rm)):
        del df[var_names_rm[i]]
    var_names_new = list(df.columns)
    return var_names_new


def est_fs(df, t0, var_names):
    time = list(df["time_dn"] - t0)
    t_start, t_end = time[0], time[-1]
    duration = t_end - t_start  # units: days
    fs = math.ceil(len(time) / duration / DAY_TO_SEC)

    for i in range(len(var_names)):
        var_name = var_names[i]
        ts = df[var_name]
        fs = math.ceil(len(ts) / duration / DAY_TO_SEC)

    return fs


def preprocess_path(path, data_root, task_label, numsecs, numsecs_overlap):
    subject_label, session, run, subject, diff_level = path

    # EDA
    modality = "lslshimmereda"
    df_EDA = load_time_series(
        data_root, subject_label, session, task_label, modality, run
    )
    EDA_vars = list(df_EDA.columns)[1:]

    # ECG
    modality = "lslshimmerecg"
    df_ECG = load_time_series(
        data_root, subject_label, session, task_label, modality, run
    )
    ECG_vars = list(df_ECG.columns)[1:]

    # Forearm Accelerometry and EMG data
    modality = "lslshimmeremg"
    df_EMG = load_time_series(
        data_root, subject_label, session, task_label, modality, run
    )
    EMG_vars = list(df_EMG.columns)[1:]

    # Torso Accelerometry
    modality = "lslshimmertorsoacc"
    df_TAC = load_time_series(
        data_root, subject_label, session, task_label, modality, run
    )
    TAC_vars = list(df_TAC.columns)[1:]

    # Respiration
    modality = "lslshimmerresp"
    df_RespSD = load_time_series(
        data_root, subject_label, session, task_label, modality, run
    )
    RespSD_vars = list(df_RespSD.columns)[1:]

    # Eye Tracking
    modality = "lslhtcviveeye"
    df_EYE = load_time_series(
        data_root, subject_label, session, task_label, modality, run
    )
    EYE_vars = list(df_EYE.columns)[1:]

    # sample conversion to wall-clock datetime
    # collect several time stamp signals and convert to epoch time
    t_eda = pd.to_datetime(df_EDA["time_dn"] - T0, unit="D")
    t_ecg = pd.to_datetime(df_ECG["time_dn"] - T0, unit="D")
    t_emg = pd.to_datetime(df_EMG["time_dn"] - T0, unit="D")
    t_tac = pd.to_datetime(df_TAC["time_dn"] - T0, unit="D")
    t_res = pd.to_datetime(df_RespSD["time_dn"] - T0, unit="D")
    t_eye = pd.to_datetime(df_EYE["time_dn"] - T0, unit="D")

    # remove non-informative EYE signals
    EYE_vars = remove_vars(df_EYE, EYE_vars, EYE_VARS_RM)

    fs_EDA = est_fs(df_EDA, T0, EDA_vars)
    fs_ECG = est_fs(df_ECG, T0, ECG_vars)
    fs_EMG = est_fs(df_EMG, T0, EMG_vars)
    fs_TAC = est_fs(df_TAC, T0, TAC_vars)
    fs_RES = est_fs(df_RespSD, T0, RespSD_vars)
    fs_EYE = est_fs(df_EYE, T0, EYE_vars)

    # resampling of signals
    FS = 256
    EDA_rs = resample_df(
        df_EDA, EDA_vars, fs_EDA, FS, T0, plots=False, nperiods=5, label="EDA"
    )
    ECG_rs = resample_df(
        df_ECG, ECG_vars, fs_ECG, FS, T0, plots=False, nperiods=1, label="ECG"
    )
    EMG_rs = resample_df(
        df_EMG, EMG_vars, fs_EMG, FS, T0, plots=False, nperiods=1, label="EMG"
    )
    TAC_rs = resample_df(
        df_TAC, TAC_vars, fs_TAC, FS, T0, plots=False, nperiods=1, label="TAC"
    )
    RespSD_rs = resample_df(
        df_RespSD, RespSD_vars, fs_RES, FS, T0, plots=False, nperiods=5, label="RES"
    )
    EYE_rs = resample_df(
        df_EYE, EYE_vars, fs_EYE, FS, T0, plots=False, nperiods=5, label="EYE"
    )

    # segment signals into (overlapping) segments
    len_time = 1e20
    x, t = EDA_rs[0]
    len_time = np.min([len_time, len(t)])
    x, t = ECG_rs[0]
    len_time = np.min([len_time, len(t)])
    x, t = EMG_rs[0]
    len_time = np.min([len_time, len(t)])
    x, t = TAC_rs[0]
    len_time = np.min([len_time, len(t)])
    x, t = RespSD_rs[0]
    len_time = np.min([len_time, len(t)])
    x, t = EYE_rs[0]
    len_time = np.min([len_time, len(t)])
    len_time = int(len_time)

    # count number of features
    F = (
        len(EDA_vars)
        + len(ECG_vars)
        + len(EMG_vars)
        + len(TAC_vars)
        + len(RespSD_vars)
        + len(EYE_vars)
    )
    # print("Number of features: ", F)

    seglen = FS * numsecs
    overlaplen = FS * numsecs_overlap
    # print("Segment length: ", seglen)
    # print("Overlap length: ", overlaplen)

    numsegs = 0
    start = 0
    while start + seglen < len_time:
        start += overlaplen
        numsegs += 1
    # print("Number of segments: ", numsegs)

    # Group all signals into a list
    TS_rs = [EDA_rs, ECG_rs, EMG_rs, TAC_rs, RespSD_rs, EYE_rs]
    VARS = [EDA_vars, ECG_vars, EMG_vars, TAC_vars, RespSD_vars, EYE_vars]
    x_all = []
    for A_rs, var_names in zip(TS_rs, VARS):
        x_list = [A_rs[i][0][:len_time] for i in range(len(var_names))]
        for j in range(len(var_names)):
            x_all.append(x_list[j])

    # obtain array matrix of segments for each feature/signal
    X = np.zeros((F, numsegs, seglen))
    for i in range(len(x_all)):
        X[i, :, :] = segment_signal(x_all[i], numsegs, seglen, overlaplen, len_time)

    # re-permute axes
    X = np.transpose(X, (1, 0, 2))
    # print("X shape: ", X.shape)

    # make label vector
    nsegs = X.shape[0]
    y = diff_level * np.ones((nsegs,))

    # make subject attribute
    s = subject * np.ones((nsegs,))

    return X, y, s, VARS


def preprocess(
    numsecs=20,
    numsecs_overlap=5,
    data_root="datasets/downloads/cogpilot/multimodal-physiological-monitoring-during-virtual-reality-piloting-tasks-1.0.0/dataPackage",
    task_label="task-ils",
    num_processes=10,
):
    directory = Path(data_root) / task_label
    subject_label_list = list(directory.iterdir())

    cnt_subject = 0
    cnt_subjectdir = 0
    cnt_run = 0

    paths_list = []

    for subject_label in subject_label_list:
        if subject_label.name[:6] == "sub-cp":
            subject = int(subject_label.name[7:9])

            if subject_label.name in SUBJ_RM:
                # Exclude runs with missing signals
                continue

            cnt_subject += 1

            for session_label in subject_label.iterdir():
                cnt_subjectdir += 1

                for run in session_label.iterdir():
                    diff_level = run.name[7]
                    paths_list.append(
                        (
                            subject_label.name,
                            session_label.name,
                            run.name,
                            subject,
                            int(diff_level),
                        )
                    )
                    cnt_run += 1

    print("No subjects: ", cnt_subject)
    print("No sessions: ", cnt_subjectdir)
    print("No runs: ", cnt_run)

    with Pool(num_processes) as p:
        # process data in parallel
        output = p.map(
            partial(
                preprocess_path,
                data_root=data_root,
                task_label=task_label,
                numsecs=numsecs,
                numsecs_overlap=numsecs_overlap,
            ),
            paths_list,
        )

    X, y, s, VARS = zip(*output)
    X_all = np.concatenate(X, axis=0)
    y_all = np.concatenate(y, axis=0)
    s_all = np.concatenate(s, axis=0)
    VARS_all = VARS[0]
    data_dict = {"X": X_all, "y": y_all, "s": s_all, "Xvars": VARS_all}
    return data_dict


def prepare(data_dict):
    np.random.seed(0)

    unique_subj = np.unique(data_dict["s"])

    # Select difficulty levels 1, 4
    idx = (data_dict["y"] == 1) | (data_dict["y"] == 4)

    X = data_dict["X"][idx]
    y = data_dict["y"][idx]
    s = data_dict["s"][idx]

    # split into train, val, test subsets based on subjects
    # 70/10/20 approximate split
    nsubj = len(unique_subj)  # 29
    nsubj_train = 21
    nsubj_val = 3
    nsubj_test = nsubj - nsubj_train - nsubj_val

    # randomize subjects
    perm = np.random.permutation(nsubj)
    unique_subj = unique_subj[perm]

    subj_train = unique_subj[:nsubj_train]
    subj_val = unique_subj[nsubj_train : nsubj_train + nsubj_val]
    subj_test = unique_subj[nsubj_train + nsubj_val :]

    def get_idx(subj, s):
        list_arr = [(s == r) * 1.0 for r in subj]
        for i in range(len(subj)):
            if i == 0:
                v = list_arr[i]
            else:
                v += list_arr[i]
        idx = v > 0.5
        return idx

    idx_train = get_idx(subj_train, s)
    idx_val = get_idx(subj_val, s)
    idx_test = get_idx(subj_test, s)

    # Obtain splits and store into Data_dict_1_4.pkl
    X_train, y_train, s_train = X[idx_train], y[idx_train], s[idx_train]
    X_val, y_val, s_val = X[idx_val], y[idx_val], s[idx_val]
    X_test, y_test, s_test = X[idx_test], y[idx_test], s[idx_test]

    data_dict_1_4 = {
        "X_train": X_train,
        "y_train": y_train,
        "s_train": s_train,
        "X_val": X_val,
        "y_val": y_val,
        "s_val": s_val,
        "X_test": X_test,
        "y_test": y_test,
        "s_test": s_test,
        "Xvars": data_dict["Xvars"],
    }
    return data_dict_1_4


def normalize(data_dict):
    Ptrain, y_train, s_train = (
        data_dict["X_train"],
        data_dict["y_train"],
        data_dict["s_train"],
    )
    Pval, y_val, s_val = data_dict["X_val"], data_dict["y_val"], data_dict["s_val"]
    Ptest, y_test, s_test = (
        data_dict["X_test"],
        data_dict["y_test"],
        data_dict["s_test"],
    )
    Xvars = data_dict["Xvars"]

    # map labels {1,4} to {0,1}
    y_train[y_train == 1], y_train[y_train == 4] = 0, 1
    y_val[y_val == 1], y_val[y_val == 4] = 0, 1
    y_test[y_test == 1], y_test[y_test == 4] = 0, 1

    # re-permute last two indices
    Ptrain = Ptrain.transpose((0, 2, 1))
    Pval = Pval.transpose((0, 2, 1))
    Ptest = Ptest.transpose((0, 2, 1))
    T, F = Ptrain[0].shape

    # obtain mean, std statistics on train-set
    def get_stats(P_tensor):
        N, T, F = P_tensor.shape
        Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
        # find mean for each variable
        mf = np.zeros((F, 1))
        stdf = np.ones((F, 1))
        eps = 1e-7
        for f in range(F):
            vals_f = Pf[f, :]
            #         # extract values on non-missing data
            #         vals_f = vals_f[vals_f>0]
            # compute mean, std
            mf[f] = np.mean(vals_f)
            stdf[f] = np.std(vals_f)
            stdf[f] = np.max([stdf[f], eps])
        return mf, stdf

    # get mean, std stats from train set
    mf, stdf = get_stats(Ptrain)

    def normalizeT(P_tensor, mf, stdf):
        """Normalize time series variables."""
        N, T, F = P_tensor.shape
        Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)

        # input normalization
        # normalize by channel
        for f in range(F):
            Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)

        Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
        return Pnorm_tensor

    def tensorize_normalize(P, y, mf, stdf):
        T, F = P[0].shape
        P_tensor = normalizeT(P, mf, stdf)
        y_tensor = y.reshape(-1, 1)
        y_tensor = y_tensor[:, 0]
        return P_tensor, y_tensor

    Ptrain_tensor, ytrain_tensor = tensorize_normalize(Ptrain, y_train, mf, stdf)
    Pval_tensor, yval_tensor = tensorize_normalize(Pval, y_val, mf, stdf)
    Ptest_tensor, ytest_tensor = tensorize_normalize(Ptest, y_test, mf, stdf)

    # convert observations to (seq_len, batch, feats)
    Ptrain_tensor = Ptrain_tensor.transpose((1, 0, 2))
    Pval_tensor = Pval_tensor.transpose((1, 0, 2))
    Ptest_tensor = Ptest_tensor.transpose((1, 0, 2))

    # put normalized data into dictionary
    data_dict = {
        "Ptrain": Ptrain_tensor,
        "Pval": Pval_tensor,
        "Ptest": Ptest_tensor,
        "ytrain": ytrain_tensor,
        "yval": yval_tensor,
        "ytest": ytest_tensor,
        "strain": s_train,
        "sval": s_val,
        "stest": s_test,
        "Vars": Xvars,
    }
    return data_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--numsecs", default=20)
    parser.add_argument("--numsecs_overlap", default=5)
    parser.add_argument("--num_processes", default=10)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()

    out_dir = Path("datasets/cogpilot")
    out_dir.mkdir(exist_ok=True)

    # 1. preprocess timeseries
    preprocess_path = (
        out_dir / f"CogPilot_MTS_{args.numsecs}_{args.numsecs_overlap}.pkl"
    )
    if not args.overwrite and preprocess_path.exists():
        print(
            f"Found preprocessed data with numsecs={args.numsecs}, numsecs_overlap={args.numsecs_overlap} at {preprocess_path}. To redo, use --overwrite"
        )
        with open(preprocess_path, "rb") as f:
            data_dict = pickle.load(f)
    else:
        print(
            f"Preprocessing data with numsecs={args.numsecs}, numsecs_overlap={args.numsecs_overlap} to {preprocess_path}"
        )
        data_dict = preprocess(numsecs=20, numsecs_overlap=5, num_processes=10)
        with open(preprocess_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=4)

    # 2. split into train/test/val, stratified by participant, using labels 1 and 4 only
    prepare_path = (
        out_dir / f"CogPilot_MTS_1_4_{args.numsecs}_{args.numsecs_overlap}.pkl"
    )
    if not args.overwrite and prepare_path.exists():
        print(
            f"Found train/val/test data (1/4 difficulty) with numsecs={args.numsecs}, numsecs_overlap={args.numsecs_overlap} at {prepare_path}. To redo, use --overwrite"
        )
        with open(prepare_path, "rb") as f:
            data_dict = pickle.load(f)
    else:
        print(
            f"Preparing train/val/test data (1/4 difficulty) with numsecs={args.numsecs}, numsecs_overlap={args.numsecs_overlap} to {prepare_path}"
        )
        data_dict = prepare(data_dict)
        with open(prepare_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=4)

    # 2. normalize data, convert labels to 0/1
    norm_path = (
        out_dir / f"CogPilot_MTS_1_4_{args.numsecs}_{args.numsecs_overlap}_norm.pkl"
    )
    if not args.overwrite and norm_path.exists():
        print(
            f"Found normalized train/val/test data (1/4 difficulty) with numsecs={args.numsecs}, numsecs_overlap={args.numsecs_overlap} at {norm_path}. To redo, use --overwrite"
        )
    else:
        print(
            f"Preparing normalized train/val/test data (1/4 difficulty) with numsecs={args.numsecs}, numsecs_overlap={args.numsecs_overlap} to {norm_path}"
        )
        data_dict = normalize(data_dict)
        with open(norm_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=4)
