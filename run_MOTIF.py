import os
import sys
import argparse

parser = argparse.ArgumentParser(
    description=("This script implements a fairness-aware deep learning model designed to mitigate\n"
                 "racial disparities in breast cancer prognosis. It integrates transfer learning and\n"
                 "synthetic data augmentation (SMOTE) with weighted multi-omics representation to\n"
                 "improve predictive performance in underrepresented populations. The model is\n"
                 "pretrained on European American data and fine-tuned on African American data using\n"
                 "contrastive domain adaptation.\n"
                ),
    epilog=(
        "Examples:\n"
        "   python run_MOTIF.py \\\n"
        "       --dat_path /home/user/project/data \\\n"
        "       --out_path /home/user/project/output \\\n"
        "       --cpoint PFI \\\n"
        "       --year 3"
        ),
    formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=80)
)

parser.add_argument(
    "--dat_path",
    type=str,
    required=True,
    help="Path to the data like multi-omics data and other materials for model training."
)
parser.add_argument(
    "--out_path",
    type=str,
    required=True,
    help="Path for all output files after model training."
)
parser.add_argument(
    "--cpoint",
    type=str,
    required=True,
    help="Clinical endpoint of interest for prognosis prediction (e.g., PFI, DSS)."
)
parser.add_argument(
    "--year",
    type=int,
    required=True,
    help="Time threshold used to define positive prognosis events."
)

args = parser.parse_args()

# Check arguments
print("Data path: ", args.dat_path)
print("Output path: ", args.out_path)
print("Clinical point: ", args.cpoint)
print("Year: ", args.year)

# Set module directory
input_path = args.dat_path
sys.path.append(input_path)

# Set working directory
folder_path = args.out_path
cf_dir_path = args.cpoint
ynum = args.year
year_path = str(ynum) + "yr"
            
if not os.path.exists(folder_path + "/" + year_path + "/" + cf_dir_path):
    os.makedirs(folder_path + "/" + year_path + "/" + cf_dir_path)
    print(year_path +  "/" + cf_dir_path + " is generated now.")
            
else:
    print(year_path + "/" + cf_dir_path + " is already there.")
os.chdir(folder_path + "/" + year_path + "/" + cf_dir_path)

if input_path in sys.path:
    print(f"Data directory: " + input_path)
else:
    print(f"Please check input directory")

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import roc_auc_score
import pickle
import Initialization
import keras
from keras import Input, Model
from keras.layers import Activation, Dense, Dropout, Lambda
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.optimizers import SGD
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from theano.tensor.shared_randomstreams import RandomStreams
from imblearn.over_sampling import SMOTE

devices = tf.config.experimental.list_physical_devices("GPU")
if devices:
    print("GPU is detected.")
    tf.config.experimental.set_memory_growth(devices[0], True)
    print("GPU dynamic memory allocation is activated.")
else:
    print("GPU can't used for it.")

def get_k_best(X_train, y_train, X_test, k_num):
    k_best = SelectKBest(f_classif, k=k_num)
    k_best.fit(X_train, y_train)
    res = (k_best.transform(X_train), k_best.transform(X_test))
    return res

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high=np.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out),
                ),
                dtype=theano.config.floatX,
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name="W", borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]

    def reset_weight(self, params):
        self.W.set_value(params[0])
        self.b.set_value(params[1])

class DropoutHiddenLayer(HiddenLayer):
    def __init__(
        self,
        rng,
        input,
        n_in,
        n_out,
        is_train,
        activation,
        dropout_rate,
        mask=None,
        W=None,
        b=None,
    ):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_out,
            W=W,
            b=b,
            activation=activation,
        )
        self.dropout_rate = dropout_rate
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.mask = mask
        self.layer_output = self.output

        train_output = self.drop(self.layer_output, self.dropout_rate)
        test_output = self.output * (1 - dropout_rate)
        self.output = ifelse(T.eq(is_train, 1), train_output, test_output)
        return

    def drop(self, input, p=0.5):
        mask = self.srng.binomial(
            n=1, p=p, size=input.shape, dtype=theano.config.floatX
        )
        return input * mask

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name="W",
            borrow=True,
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name="b",
            borrow=True,
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        # the output of the softmax layer.
        self.output = T.nnet.softmax(T.dot(input, self.W) + self.b)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                "y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type),
            )
        # check if y is of the correct datatype
        if y.dtype.startswith("int"):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def reset_weight(self, params):
        self.W.set_value(params[0])
        self.b.set_value(params[1])

class dA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        non_lin=None,
        ce=False,
    ):
        self.non_lin = non_lin
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.ce = ce
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

            # note : W' was written as `W_prime` and b' as `b_prime`
            if not W:
                # W is initialized with `initial_W` which is uniformely sampled
                # from -4*sqrt(6./(n_visible+n_hidden)) and
                # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
                # converted using asarray to dtype
                # theano.config.floatX so that the code is runable on GPU
                initial_W = np.asarray(
                    numpy_rng.uniform(
                        low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                        high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                        size=(n_visible, n_hidden),
                    ),
                    dtype=theano.config.floatX,
                )
                W = theano.shared(value=initial_W, name="W", borrow=True)

            if not bvis:
                bvis = theano.shared(
                    value=np.zeros(n_visible, dtype=theano.config.floatX),
                    borrow=True,
                )

            if not bhid:
                bhid = theano.shared(
                    value=np.zeros(n_hidden, dtype=theano.config.floatX),
                    name="b",
                    borrow=True,
                )

            self.W = W
            # b corresponds to the bias of the hidden
            self.b = bhid
            # b_prime corresponds to the bias of the visible
            self.b_prime = bvis
            # tied weights, therefore W_prime is W transpose
            self.W_prime = self.W.T
            self.theano_rng = theano_rng
            # if no input is given, generate a variable representing the input
            if input is None:
                # we use a matrix because we expect a minibatch of several
                # examples, each example being a row
                self.x = T.dmatrix(name="input")
            else:
                self.x = input

            self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return (
            self.theano_rng.binomial(
                size=input.shape,
                n=1,
                p=1 - corruption_level,
                dtype=theano.config.floatX,
            )
            * input
        )

    def get_hidden_values(self, input):
        # Computes the values of the hidden layer
        return self.non_lin((T.dot(input, self.W) + self.b))

    def get_reconstructed_input(self, hidden):
        # Computes the reconstructed input given the values of the hidden layer
        return self.non_lin((T.dot(hidden, self.W_prime) + self.b_prime))

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        if self.ce:
            L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        else:
            L = T.sum((self.x - z) ** 2, axis=1)
            
        cost = T.mean(L)

        # Compute the gradients of the cost of the `dA` with respect to its parameters
        gparams = T.grad(cost, self.params)
        
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)

def standarize_dataset(data):
    X = data["X"]
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_new["X"] = X
    return data_new

def normalize_dataset(data):
    X = data["X"]
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    X = preprocessing.normalize(X)
    data_new["X"] = X
    return data_new

def get_race(cancer_type):
    path = input_path + "/Genetic_Ancestry.xlsx"
    df_list = [
        pd.read_excel(
            path, disease, usecols="A,E", index_col="Patient_ID", keep_default_na=False
        )
        for disease in [cancer_type]
    ]
    df_race = pd.concat(df_list)
    df_race = df_race[df_race["EIGENSTRAT"].isin(["EA", "AA", "EAA", "NA", "OA"])]
    df_race["race"] = df_race["EIGENSTRAT"]

    df_race.loc[df_race["EIGENSTRAT"] == "EA", "race"] = "WHITE"
    df_race.loc[df_race["EIGENSTRAT"] == "AA", "race"] = "BLACK"
    df_race.loc[df_race["EIGENSTRAT"] == "EAA", "race"] = "ASIAN"
    df_race.loc[df_race["EIGENSTRAT"] == "NA", "race"] = "NAT_A"
    df_race.loc[df_race["EIGENSTRAT"] == "OA", "race"] = "OTHER"
    df_race = df_race.drop(columns=["EIGENSTRAT"])

    return df_race

def add_race_CT(cancer_type, df, target, groups):
    df_race = get_race(cancer_type)
    df_race = df_race[df_race["race"].isin(groups)]
    df_C_T = get_CT(target)

    # Keep patients with race information
    df = df.join(df_race, how="inner")
    print(df.shape)
    df = df.dropna(axis="columns")
    df = df.join(df_C_T, how="inner")
    print(df.shape)

    # Packing the data
    C = df["C"].tolist()
    R = df["race"].tolist()
    T = df["T"].tolist()
    E = [1 - c for c in C]
    df = df.drop(columns=["C", "race", "T"])
    X = df.values
    X = X.astype("float32")
    data = {
        "X": X,
        "T": np.asarray(T, dtype=np.float32),
        "C": np.asarray(C, dtype=np.int32),
        "E": np.asarray(E, dtype=np.int32),
        "R": np.asarray(R),
        "Samples": df.index.values,
        "FeatureName": list(df),
    }

    return data

def get_one_race(dataset, race):
    X, T, C, E, R = dataset["X"], dataset["T"], dataset["C"], dataset["E"], dataset["R"]
    mask = R == race
    X, T, C, E, R = X[mask], T[mask], C[mask], E[mask], R[mask]
    data = {"X": X, "T": T, "C": C, "E": E, "R": R}
    return data

def get_CT(target):
    path1 = input_path + "/TCGA_CDR.xlsx"
    if target == "DSS":
        cols = "B,Z,AA"
    elif target == "PFI":
        cols = "B,AB,AC"

    df_C_T = pd.read_excel(
        path1, "TCGA-CDR", usecols=cols, index_col="bcr_patient_barcode"
    )
    df_C_T.columns = ["E", "T"]
    df_C_T = df_C_T[df_C_T["E"].isin([0, 1])]
    df_C_T = df_C_T.dropna()
    df_C_T["C"] = 1 - df_C_T["E"]
    df_C_T.drop(columns=["E"], inplace=True)
    return df_C_T

def get_n_years(dataset, years):
    X, T, C, E, R = dataset["X"], dataset["T"], dataset["C"], dataset["E"], dataset["R"]

    df = pd.DataFrame(X)
    df["T"] = T
    df["C"] = C
    df["R"] = R
    df["Y"] = 1

    df = df[~((df["T"] < 365 * years) & (df["C"] == 1))]
    df.loc[df["T"] <= 365 * years, "Y"] = 0
    df["strat"] = df.apply(lambda row: str(row["Y"]) + str(row["R"]), axis=1)
    df = df.reset_index(drop=True)

    R = df["R"].values
    Y = df["Y"].values
    y_strat = df["strat"].values
    df = df.drop(columns=["T", "C", "R", "Y", "strat"])
    X = df.values
    y_sub = R  # doese not matter

    return (X, Y.astype("int32"), R, y_sub, y_strat)

def train_and_predict(
    X_train_target,
    y_train_target,
    X_train_source,
    y_train_source,
    X_val_target,
    Y_val_target,
    X_test,
    y_test,
    repetition,
    sample_per_class,
    alpha=0.25,
    learning_rate=0.01,
    hiddenLayers=[100, 50],
    dr=0.5,
    momentum=0.0,
    decay=0,
    batch_size=32,
    n_features=400,
):
    # size of input variable for each patient
    domain_adaptation_task = "WHITE_to_BLACK"
    input_shape = (n_features,)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # number of classes for digits classification
    nb_classes = 2
    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = alpha

    # Having two streams. One for source and one for target.
    model1 = Initialization.Create_Model(hiddenLayers=hiddenLayers, dr=dr)
    processed_a = model1(input_a)
    processed_b = model1(input_b)

    # Creating the prediction function. This corresponds to h in the paper.
    processed_a = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(processed_a)
    out1 = Activation("softmax", name="classification")(out1)

    distance = Lambda(
        Initialization.euclidean_distance,
        output_shape=Initialization.eucl_dist_output_shape,
        name="CSA",
    )([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=learning_rate, momentum=momentum
    )  # momentum=0., decay=0., decay=decay
    model.compile(
        loss={
            "classification": "binary_crossentropy",
            "CSA": Initialization.contrastive_loss,
        },
        optimizer=optimizer,
        loss_weights={"classification": 1 - alpha, "CSA": alpha},
    )

    print("Domain Adaptation Task: " + domain_adaptation_task)
    # for repetition in range(10):
    Initialization.Create_Pairs(
        domain_adaptation_task,
        repetition,
        sample_per_class,
        X_train_target,
        y_train_target,
        X_train_source,
        y_train_source,
        n_features=n_features,
    )
    best_score, best_Auc = Initialization.training_the_model(
        model,
        domain_adaptation_task,
        repetition,
        sample_per_class,
        batch_size,
        X_val_target,
        Y_val_target,
        X_test,
        y_test,
    )

    print(
        "Best AUC for {} target sample per class and repetition {} is {}.".format(
            sample_per_class, repetition, best_Auc
        )
    )
    return best_score, best_Auc

def run_CCSA_transfer_SMOTE(
    seed,
    dataset,
    n_features,
    fold=3,
    alpha=0.25,
    learning_rate=0.01,
    hiddenLayers=[100, 50],
    dr=0.5,
    groups=("WHITE", "BLACK"),
    momentum=0.0,
    decay=0,
    batch_size=32,
    sample_per_class=2,
    repetition=1,
):
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(X)
    df["R"] = R
    df["Y"] = Y

    # WHITE 그룹 (소스 도메인)과 BLACK 그룹 (타겟 도메인) 분리
    df_train = df[df["R"] == groups[0]]  # WHITE 그룹 (훈련 데이터)
    df_w_y = df_train["Y"]
    df_train = df_train.drop(columns=["Y", "R"])

    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values

    df_test = df[df["R"] == groups[1]]  # BLACK 그룹 (타겟 데이터)
    df_b_y = df_test["Y"]
    df_test = df_test.drop(columns=["Y", "R"])

    Y_test = df_b_y.values.ravel()
    X_test = df_test.values

    print("NumFeature: " + str(n_features))

    if n_features > 0 and n_features < X_test.shape[1]:
        X_train_source, X_test = get_k_best(
            X_train_source, Y_train_source, X_test, k_num=n_features
        )
    else:
        n_features = X_test.shape[1]

    df_score = pd.DataFrame(columns=["scr", "Y", "pred"])
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]

        # 1. SMOTE를 타겟 도메인 (BLACK 그룹) 훈련 데이터에 적용
        smote = SMOTE(sampling_strategy = "auto",
                      n_jobs=-1,
                      random_state=33,
                      k_neighbors = 5)
        X_train_target_resampled, Y_train_target_resampled = smote.fit_resample(X_train_target_full, Y_train_target_full)

        # 2. 타겟 도메인 데이터에서 일부 샘플을 검증 데이터로 분리
        index0 = np.where(Y_train_target_resampled == 0)
        index1 = np.where(Y_train_target_resampled == 1)

        target_samples = []
        target_samples.extend(index0[0][0:sample_per_class])
        target_samples.extend(index1[0][0:sample_per_class])

        X_train_target = X_train_target_resampled[target_samples]
        Y_train_target = Y_train_target_resampled[target_samples]

        X_val_target = [
            e for idx, e in enumerate(X_train_target_full) if idx not in target_samples
        ]
        Y_val_target = [
            e for idx, e in enumerate(Y_train_target_full) if idx not in target_samples
        ]
        X_val_target = np.array(X_val_target)
        Y_val_target = np.array(Y_val_target)

        best_score, best_Auc = train_and_predict(
            X_train_target,
            Y_train_target,
            X_train_source,
            Y_train_source,
            X_val_target,
            Y_val_target,
            X_test_target,
            Y_test_target,
            sample_per_class=sample_per_class,
            alpha=alpha,
            learning_rate=learning_rate,
            hiddenLayers=hiddenLayers,
            dr=dr,
            momentum=momentum,
            decay=decay,
            batch_size=batch_size,
            repetition=repetition,
            n_features=n_features,
        )

        print(best_score.shape)
        print(Y_test_target.shape)
        pred = (best_score > 0.5).astype(int)
        array = np.column_stack((best_score, Y_test_target, pred))
        df_temp = pd.DataFrame(
            array, index=list(test_index), columns=["scr", "Y", "pred"]
        )
        df_score = df_score.append(df_temp)

    # ROC-AUC
    roc_auc = roc_auc_score(df_score["Y"].values, df_score["scr"].values)

    res = {"TL_auc": roc_auc}
    df = pd.DataFrame(res, index=[seed])
    
    return df, df_score["scr"].values

def run_BRCA_inter_cv(data, years, target, k_value, fn, tn, comb1, comb2, comb3):
    print("The number of feature: " + str(k_value))
    dataset = data
    if dataset["X"].shape[0] < 10:
        return None
    dataset = standarize_dataset(dataset)
    dataset_w = get_one_race(dataset, "WHITE")

    dataset_w = get_n_years(dataset_w, years)
    dataset_b = get_one_race(dataset, "BLACK")

    dataset_b = get_n_years(dataset_b, years)

    dataset_tl = normalize_dataset(dataset)
    dataset_tl = get_n_years(dataset_tl, years)

    dataset = get_n_years(dataset, years)

    k = k_value
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=["RY"])
    df["R"] = R
    df["Y"] = Y
    print(X.shape)
    Dict = df["RY"].value_counts()

    Dict = dict(Dict)
    print(Dict)
    for key in Dict:
        print(key, Dict[key])
    
    parameters_CCSA = {
        "fold": fn,
        "n_features": k,
        "alpha": 0.3,
        "batch_size": 20,
        "learning_rate": 0.01,
        "hiddenLayers": [100],
        "dr": 0.0,
        "momentum": 0.9,
        "decay": 0.0,
        "sample_per_class": 2,
    }

    res = pd.DataFrame()
    score_dict = {}
    for i in range(tn):
        seed = i
        s_time = time.time()
        df_tl, s_ccsa = run_CCSA_transfer_SMOTE(seed, dataset_tl, **parameters_CCSA)
        e_time = time.time()
        n_time = e_time - s_time
        nm_time = n_time / 60
        print(f"N{i}", f"DomainAdaptation model: {nm_time:.2f}")
        print(df_tl.to_string(index=False))
        df1 = pd.concat(
            [
                df_tl,
            ],
            sort=False,
            axis=1,
        )
        s_ccsa_df = pd.DataFrame(s_ccsa)
        all_s = pd.concat(
            [s_ccsa_df],
            sort=False,
            axis=1,
        )
        res = res.append(df1)
        score_dict[i] = all_s

    fkey = str(comb1) + "_" + str(comb2) + "_" + str(comb3)

    f_name = "BRCA-AA-EA-TripleOmics-MOTIFres-" + str(target) + "-" + str(years) + "YR_MJ_K" + str(k) + "_" + fkey + ".xlsx"
    res.to_excel(f_name)
    summary_df = pd.DataFrame(
        {"Column": res.columns, "Mean": res.mean(), "Standard Deviation": res.std()}
    )
    summary_df.to_excel(
        "summary-BRCA-AA-EA-TripleOmics-MOTIFres-" + str(target) + "-" + str(years) + "YR_MJ_K" + str(k) + ".xlsx"
    )
    name = "BRCA-AA-EA-TripleOmics-MOTIFres-" + str(target) + "-" + str(years) + "YR_MJ_K" + str(k) + "_" + fkey + ".pkl"
    with open(name, "wb") as file:
        pickle.dump(score_dict, file)

# Run script with RNA_exp file.
DF = pd.read_csv(input_path + "/BRCA_RNA.csv")
DF = DF.T
DF.columns = DF.iloc[0]
DF = DF[1:]
df = DF
new_index = [str(index)[:12] for index in df.index]
df.index = new_index
df = df.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")

DF1 = pd.read_csv(input_path + "/BRCA_miRNA.csv")
DF1 = DF1.T
DF1.columns = DF1.iloc[0]
DF1 = DF1[1:]
df1 = DF1
new_index1 = [str(index)[:12] for index in df1.index]
df1.index = new_index1
df1 = df1.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")

DF2 = pd.read_csv(input_path + "/BRCA_Methyl.csv")
DF2 = DF2.T
DF2.columns = DF2.iloc[0]
DF2 = DF2[1:]
df2 = DF2
new_index2 = [str(index)[:12] for index in df2.index]
df2.index = new_index2
df2 = df2.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
df2.rename(columns={'T': 'T_1'}, inplace=True)

df_df1_intersection = np.intersect1d(df.index, df1.index)
fin_intersection = np.intersect1d(df_df1_intersection, df2.index)

df = df.loc[fin_intersection,:]
df1 = df1.loc[fin_intersection,:]
df2 = df2.loc[fin_intersection,:]
df_m = df.T
df_m = df_m.to_numpy()
df_m = df_m.astype("float32")
df_mi = df1.T
df_mi = df_mi.to_numpy()
df_mi = df_mi.astype("float32")
df_me = df2.T
df_me = df_me.to_numpy()
df_me = df_mi.astype("float32")

cm_m = np.corrcoef(df_m, rowvar=False)
cm_mi = np.corrcoef(df_mi, rowvar=False)
cm_me = np.corrcoef(df_me, rowvar=False)

start_num = 0.1
end_num = 1.0
step_size = 0.1

weight_nums = np.arange(start_num, end_num, step_size)
weight_list = np.round(weight_nums, 1)

combinations = []
for x in range(len(weight_list)):
    opp_weight_list1 = np.round(np.arange(start_num, 1-weight_list[x], step_size), 1)
    opp_weight_list2 = np.round(1-(opp_weight_list1+weight_list[x]), 1)
    opp_weight_list0 = [weight_list[x]] * len(opp_weight_list1)
    combinations = combinations + list(zip(opp_weight_list0, opp_weight_list1, opp_weight_list2))

for x in range(0, len(combinations)):
    wc01 = combinations[x][0]
    wc02 = combinations[x][1]
    wc03 = combinations[x][2]
    print("***** Start: (" + str(wc01) + ", " + str(wc02) + ", " + str(wc03) + ") *****") 
    inter_m_mi = wc01*cm_m + wc02*cm_mi + wc03*cm_me
    inter_m_mi = pd.DataFrame(inter_m_mi)
    inter_m_mi.index = df.index
    inter_m_mi_BRCA = add_race_CT(cancer_type = 'BRCA', df = inter_m_mi, target = cf_dir_path, groups = ("WHITE", "BLACK"))

    print("------------ Start: (" + str(ynum) + "yr_" + cf_dir_path + "): " + str(700) + " ------------")
    run_BRCA_inter_cv(data=inter_m_mi_BRCA, years=ynum, target=cf_dir_path, k_value=700,
                      fn = 10, tn = 5, comb1 = wc01, comb2 = wc02, comb3 = wc03)
    print("------------ Finish: (" + str(ynum) + "yr_" + cf_dir_path + "): " + str(700) + " ------------")
    print("***** Finish: (" + str(wc01) + ", " + str(wc02) + ", " + str(wc03) + ") *****") 
        
print("================= All trainings are over. =================")
