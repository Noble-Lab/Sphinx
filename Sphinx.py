# Do imports 
import numpy as np
import torch
import cooler as cooler
import pickle
import os
import scipy
import scipy.stats
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import copy
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

class hicData():
    """
    A class used to hold contact map data, such as HiC and convert from .mcool files

    
    Attributes
    __________
    folder: str
        A string that identifies where the .mcool files are stored
    metadata_train: pd.DataFrame
        A pandas dataframe that contains information about the files for training data examples with columns 'Biosource', 'Assay Type', 'ExperimentSetAccession', 'ProcessedFileHref', 'Total Count', 'Log Total Count'
    metadata_valid: pd.DataFrame
        Same formatting as metadata_train except for files in the validation data set
    metadata_test: pd.DataFrame
        Same formatting as metadata_train except for files in the test data set
    resolution: int
        Resolution of the data to be loaded
    chromosome: int
        Chromosome of the data to be loaded. Currently only one chromosome can be loaded at a time. 
    scale: int
        The total number of counts 
    extension: str
        The extension of the files in metadata

    """
    def __init__(self, folder, metadata_train, metadata_valid, metadata_test, resolution, chromosome, scale, extension=".mcool"):
        self.folder = folder
        self.metadata_train = metadata_train
        self.metadata_valid = metadata_valid
        self.metadata_test = metadata_test
        self.resolution = resolution
        self.chromosome = chromosome
        self.scale = scale
        
        train_names = [f"{Path(x).stem}{extension}" for x in metadata_train["ProcessedFileHref"]]
        valid_names = [f"{Path(x).stem}{extension}" for x in metadata_valid["ProcessedFileHref"]]
        test_names = [f"{Path(x).stem}{extension}" for x in metadata_test["ProcessedFileHref"]]
        
        self.train_data = self.get_matrices_unpruned(train_names)
        self.whitelist = hicData.get_whitelist(self.train_data)
        
        self.train_data = self.get_matrices(train_names)
        self.valid_data = self.get_matrices(valid_names)
        self.test_data = self.get_matrices(test_names)
        
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)
        
        self.hic_dim = self.train_data[0].shape[0]
        
        metadata = pd.concat([metadata_train, metadata_valid, metadata_test])
        self.celltype_dict = self.get_dictionary(metadata["Biosource"])
        self.assaytype_dict = self.get_dictionary(metadata["Assay Type"])
        
    def get_dictionary(self, labels):
        """
        Makes a dictionary that makes labels to a dictionary with integer labels


        Parameters
        __________
        labels: list
            List of labels that need to have associated integer codes


        Returns
        _______
        dictionary:
            takes in label and returns integer valued code
        """
        unique_labels = np.unique(labels)
        return({label: i for i, label in enumerate(unique_labels)})
        
    def get_matrices_unpruned(self, file_names):
        """
        Uses the Cooler package to read in cooler files from a list of file names from self.folder

        Parameters
        __________
        file_names: list of str
            List of names to read in from the cooler files. If the extension is .mcool, then we include the resolution for reading .mcool files. Otherwise, resolution is not used. 


        Returns
        _______
        list:
            List of coolers that have been converted into sparse matrices and transformed using the log(x+1) transformation
        """
        extensions = np.array([Path(fn).suffix for fn in file_names])
        if sum(extensions == extensions[0]) < len(extensions) :
            raise ValueError("All extensions must be the same")
        extension = extensions[0]
        if extension == ".mcool" :
            coolers = [cooler.Cooler(f"{self.folder}{fn}::/resolutions/{self.resolution}") for fn in file_names]
        elif extension == ".cool":
            coolers = [cooler.Cooler(f"{self.folder}{fn}") for fn in file_names]
        else:
            raise ValueError("Extension must be either .cool or .mcool")
        coolers = hicData.get_sparse_matrices(coolers, self.chromosome)
        coolers = hicData.log_plus_one_matrices(coolers)
        return(coolers)
    
    def get_matrices(self, file_names):
        """
        Gets matrices from a list of file names, and prunes them

        Parameters
        __________
        file_names: list of str
            List of names to read in from the cooler files. If the extension is .mcool, then we include the resolution for reading .mcool files. Otherwise, resolution is not used. 

        Returns: 
        ________
        list:
            List of coolers that have been sparsified, transformed with the log(x+1) transformation, pruned, and normalized.
        """
        extensions = np.array([Path(fn).suffix for fn in file_names])
        if sum(extensions == extensions[0]) < len(extensions) :
            raise ValueError("All extensions must be the same")
        extension = extensions[0]
        if extension == ".mcool" :
            coolers = [cooler.Cooler(f"{self.folder}{fn}::/resolutions/{self.resolution}") for fn in file_names]
        elif extension == ".cool":
            coolers = [cooler.Cooler(f"{self.folder}{fn}") for fn in file_names]
        else:
            raise ValueError("Extension must be either .cool or .mcool")
        coolers = hicData.get_sparse_matrices(coolers, self.chromosome)
        coolers = hicData.log_plus_one_matrices(coolers)
        coolers = hicData.prune(coolers, self.whitelist)
        coolers = hicData.normalize_matrices(coolers, self.scale)
        return(coolers)
    
    def find_celltype_assay(self, celltype, assay):
        """
        Finds the matrix associated with a given celltype and assay along with what data partition it was in

        Parameters:
        ___________
        celltype: str
            The celltype to look for
        assay: str
            The assay to look for

        Returns:
        ________
        np.array:
            The observed data 
        str:
            name of the data partition it was located in. 
        """
        obs = None
        metadatas = [self.metadata_train,
                     self.metadata_valid,
                     self.metadata_test]
        datas = [self.train_data, 
                 self.valid_data,
                 self.test_data]
        names = ["train", "valid", "test"]
        done = False
        ret_obs = None
        ret_name = None
        for metadata, data, name in zip(metadatas, datas, names):
            n = metadata.shape[0]
            mats = [i for i in range(n) if ((metadata["Assay Type"][i] == assay) and (metadata["Biosource"][i] == celltype))]
            if len(mats) > 1:
                raise ValueError("Only one match allowed in metadata")
            if len(mats) == 1 and done:
                raise ValueError("Multiple matches in different metadatas")
            if len(mats) == 1:
                done = True
                obs_idx = mats[0]
                ret_obs = data[obs_idx]
                ret_name = name
        return ret_obs, ret_name
            
    def get_contact_profile(mat):
        """
        Gets the contact decay profile from a matrix by taking the average value along each of the diagonals of the matrix

        Parameters:
        ___________
        mat: np.array
            Matrix to take the contact decay profile of

        Returns:
        ________
        np.array: 
            An array containing the values of the contact decay profile
        """

        start = time.time()
        assert(mat.shape[0] == mat.shape[1])
        n = mat.shape[0]
        avgs = np.zeros(n)
        for i in range(n):
            avgs[i] = np.mean(mat.diagonal(i))
        return(avgs)
    
    def make_cdp(self, model, celltype, assay, ax=None, xmin=None, xmax = None, ymin=None, ymax=None):
        """
        Creates a plot with the contact decay profile of the contact map

        Parameters:
        ___________
        model: DeepMatrixFactorization
            A model to predict a contact map for the given celltype and assay
        celltype: str
            What celltype to make the contact decay profile for
        assay: str
            What assay to make the contact decay profile for

        Returns:
        ________
        ax:
            Matplotlib axis containing the plot
        """
        obs, metadata_name = self.find_celltype_assay(celltype, assay)
        mean_mat = m.mean_model_tensor_cross(celltype, assay).todense().A
        pred_resid = self.plot_matrix(celltype, assay).cpu().detach().numpy()
        pred_mat = pred_resid + mean_mat
        new_ax = False
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            new_ax = True
        cdp_mean = hicData.get_contact_profile(mean_mat)
        cdp_pred = hicData.get_contact_profile(pred_mat)

        if obs is not None: 
            cdp_obs = hicData.get_contact_profile(obs)
            ax.plot(cdp_obs, label="Observed", c="C0")
        ax.plot(cdp_mean, label="Mean Model", c="C1")
        ax.plot(cdp_pred, label="Predicted", c="C2")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        if new_ax:
            fig.legend(loc="lower left")
            ax.set_title(f"{celltype} {assay}")
        return(ax)
    
    def get_eigenvector(obs):
        """
        Gets the eigenvector from a matrix

        Parameters:
        ___________
        obs: np.array
            Matrix to take the eigenvector

        Returns:
        ________
        np.array: 
            An array containing the values of the eigenvector
        """
        return np.linalg.eig(obs)[1][0] #returns first eigenvector
    
    def make_eigenvector(self, model, celltype, assay, ax=None, xmin=None, xmax = None, ymin=None, ymax=None):
        """
        Creates a plot with the eigenvector of the contact map

        Parameters:
        ___________
        model: DeepMatrixFactorization
            A model to predict a contact map for the given celltype and assay
        celltype: str
            What celltype to make the contact decay profile for
        assay: str
            What assay to make the contact decay profile for

        Returns:
        ________
        ax:
            Matplotlib axis containing the plot
        """
        obs, metadata_name = self.find_celltype_assay(celltype, assay)
        mean_mat = m.mean_model_tensor_cross(celltype, assay).todense().A
        pred_resid = self.plot_matrix(celltype, assay).cpu().detach().numpy()
        pred_mat = pred_resid + mean_mat
        new_ax = False
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            new_ax = True
        cdp_mean = hicData.get_eigenvector(mean_mat)
        cdp_pred = hicData.get_eigenvector(pred_mat)

        if obs is not None: 
            cdp_obs = hicData.get_eigenvector(obs.todense().A)
            ax.plot(cdp_obs, label="Observed", c="C0")
        ax.plot(cdp_mean, label="Mean Model", c="C1")
        ax.plot(cdp_pred, label="Predicted", c="C2")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        if new_ax:
            fig.legend(loc="lower left")
            ax.set_title(f"{celltype} {assay}")
        return(ax)
    
    def get_insulation(mat, window_size=30):
        """
        Gets the insulation score from a matrix

        Parameters:
        ___________
        obs: np.array
            Matrix to take the insulation score
        window_size: int
            Size of the window to take the sum of contacts within that distance

        Returns:
        ________
        np.array: 
            An array containing the values of the insulation score
        """
        insulation = np.zeros(mat.shape[0] - 2 * window_size + 1)
        for i in range(mat.shape[0] - 2 * window_size + 1):
            Y = mat[i:(i + window_size), :]
            Y = Y[:, (i+window_size):(i + 2*window_size)]
            insulation[i] = np.mean(Y) 
        return(insulation)
    
    def make_insulation(self, model, celltype, assay, windowsize=30, ax=None, xmin=None, xmax = None, ymin=None, ymax=None):
        """
        Creates a plot with the insulation score of the contact map

        Parameters:
        ___________
        model: DeepMatrixFactorization
            A model to predict a contact map for the given celltype and assay
        celltype: str
            What celltype to make the insulation score for
        assay: str
            What assay to make the insulation score for

        Returns:
        ________
        ax:
            Matplotlib axis containing the plot
        """

        obs, metadata_name = self.find_celltype_assay(celltype, assay)
        mean_mat = m.mean_model_tensor_cross(celltype, assay).todense().A
        pred_resid = model.plot_matrix(celltype, assay).cpu().detach().numpy()
        pred_mat = pred_resid + mean_mat
        new_ax = False
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            new_ax = True
        insulation_obs = None
        if obs is not None:
            insulation_obs = hicData.get_insulation(obs.A, windowsize)
        insulation_mean = hicData.get_insulation(mean_mat, windowsize)
        insulation_pred = hicData.get_insulation(pred_mat, windowsize)
        if obs is not None:
            ax.plot(np.arange(len(insulation_obs)) * self.resolution, insulation_obs, c="C0")
        ax.plot(np.arange(len(insulation_mean)) * self.resolution, insulation_mean, c="C1")
        ax.plot(np.arange(len(insulation_pred)) * self.resolution, insulation_pred, c="C2")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        return(ax, insulation_mean, insulation_pred, insulation_obs)

    def get_sparse_matrices(coolers, chromosome):
        return [c.matrix(sparse = True, balance = False).fetch(f"chr{chromosome}").tocsr() for c in coolers]
    
    def log_plus_one_matrices(coolers):
        for c in coolers:
            c.data = np.log10(c.data + 1)
        return(coolers)
    
    def mad_max(mat, mad_cutoff) :
        marginal = mat.sum(0).A
        mad = scipy.stats.median_abs_deviation(marginal.flatten())
        med = np.median(marginal.flatten())
        mad_keep = np.logical_and(marginal < med + mad_cutoff * mad, 
                      marginal > med - mad_cutoff * mad)
        mad_keep = np.argwhere(mad_keep.flatten())
        return(mad_keep.flatten())
    
    def get_whitelist(train_data):
        mad_max_cutoff = 10
        whitelist = {}
        accepted_pos = []
        for experiment in range(len(train_data)):
            accepted_pos.append(hicData.mad_max(train_data[experiment], mad_max_cutoff))
        final_res = accepted_pos[0]
        for pos in accepted_pos:
            final_res = np.intersect1d(final_res, pos)
        return(final_res)
    
    def prune(coolers, whitelist):
        for i in range(len(coolers)):
            coolers[i] = coolers[i][:, whitelist]
            coolers[i] = coolers[i][whitelist, :]
        return coolers
    
    def normalize_matrices(matrices, scale):
        return [c / c.sum() * scale for c in matrices]
    

class hicDataset(torch.utils.data.Dataset):
    """
    Generator for contact map data that inherits from torch.utils.data.Dataset

    Attributes:
    ___________
    hicData: hicData
        A hicData object to generate data from
    split: str
        Either 'train', 'valid', or 'test', which determines which split to generate examples from
    batchsize: int
        How many examples to provide in each batch    
    fixed_idxs: 
        a list of indexes where [0] is the first index, and [1] is the second index. The generator will take examples from fixed_idxs each time, rather than generating random new indexes.
    residual: bool
        if true, then the mean model is subtracted from the observations. 
        if false, then raw values from the matrices are provided
    """
    def __init__(self, hicData, split, batchsize=10000, fixed_idxs=None, residual=False):
        self.hicData = hicData
        if split == "train":
            self.data = self.hicData.train_data
            self.n_data = self.hicData.n_train
            self.metadata = self.hicData.metadata_train
        elif split == "valid":
            self.data = self.hicData.valid_data
            self.n_data = self.hicData.n_valid
            self.metadata = self.hicData.metadata_valid
        elif split == "test":
            self.data = self.hicData.test_data
            self.n_data = self.hicData.n_test
            self.metadata = self.hicData.metadata_test
        else :
            raise ValueError("Split must be either train, valid, or test")
        if residual:
            m = MeanModel(self.hicData)
            mean_dict = copy.deepcopy(m.get_mean_model_dictionary())
            self.data = copy.deepcopy(self.data)
            for i in range(len(self.data)):
                celltype = self.metadata["Biosource"].iloc[i]
                assay = self.metadata["Assay Type"].iloc[i]
                self.data[i] = self.data[i] - mean_dict[celltype, assay]
        self.hic_dim = self.hicData.hic_dim
        self.batchsize = batchsize
        self.fixed_idxs = fixed_idxs
    
    def generate_non_diagonal_elements(self, per_exp_batch):
        """
        Generates non-diagonal elements to train the model on.

        Parameters:
        ___________
        per_exp_batch:int
            Determines how many examples are chosen to return

        """
        big_batch = int(per_exp_batch * 1.1)
        while True:
            # Make a draw that is greater than 0
            idx2_tmp = np.random.randint(0, self.hic_dim, big_batch)
            idx1_tmp = np.random.randint(0, self.hic_dim, big_batch)
            idx1 = idx1_tmp[idx1_tmp != idx2_tmp]
            idx2 = idx2_tmp[idx1_tmp != idx2_tmp]
            if len(idx1) > per_exp_batch:
                idx1 = idx1[:per_exp_batch]
                idx2 = idx2[:per_exp_batch]
                return idx1, idx2
            else:
                big_batch = int(big_batch * 1.1)
                
    
    def __getitem__(self, idx):
        per_exp_batch = np.ceil(self.batchsize / self.n_data).astype(int)
        if self.fixed_idxs is not None:
            end = min(len(self.fixed_idxs[0]), (idx + 1) * per_exp_batch)
            idx1 = self.fixed_idxs[0][(idx * per_exp_batch) : end]
            idx2 = self.fixed_idxs[1][(idx * per_exp_batch) : end]  
            per_exp_batch = len(idx1)
        idx1s = torch.zeros(per_exp_batch * self.n_data).int()
        idx2s = torch.zeros(per_exp_batch * self.n_data).int()
        celltypes = torch.zeros(per_exp_batch * self.n_data).int()
        assays = torch.zeros(per_exp_batch * self.n_data).int()
        counts = torch.zeros(per_exp_batch * self.n_data)
        y_means = torch.zeros(per_exp_batch * self.n_data)
        for i in range(self.n_data):
            if self.fixed_idxs is None:
                idx1, idx2 = self.generate_non_diagonal_elements(per_exp_batch)
            count = self.data[i][idx1, idx2]
            celltype = self.metadata["Biosource"].iloc[i]
            celltype = self.hicData.celltype_dict[celltype]
            assay = self.metadata["Assay Type"].iloc[i]
            assay = self.hicData.assaytype_dict[assay]
            chrom = 0
            idx1s[(per_exp_batch * i): (per_exp_batch * (i + 1))] = torch.tensor(idx1)
            idx2s[(per_exp_batch * i): (per_exp_batch * (i + 1))] = torch.tensor(idx2)
            counts[(per_exp_batch * i): (per_exp_batch * (i + 1))] = torch.tensor(count)
            celltypes[(per_exp_batch * i): (per_exp_batch * (i + 1))] = torch.tensor(celltype)
            assays[(per_exp_batch * i): (per_exp_batch * (i + 1))] = torch.tensor(assay)
        x = torch.vstack([celltypes, assays, idx1s, idx2s]).T
        return x, counts
    
    def __len__(self):
        if self.fixed_idxs is None:
            return np.ceil(self.hic_dim * self.hic_dim * self.n_data / self.batchsize).astype(int)
        else:
            per_exp_batch = np.ceil(self.batchsize / self.n_data).astype(int)
            return np.ceil(len(self.fixed_idxs[0]) / per_exp_batch).astype(int)

class MeanModel():
    """
    Class for the cross-mean model. 

    Attributes"
    ___________
    data: hicData
        A hicData object that is used to contain the data for the mean model
    """
    def __init__(self, data):
        self.data = data
    
    def mean_model_tensor_cross(self, celltype, assay) :
        """
        Finds the cross average prediction given a celltype and an assay

        Parameters:
        ___________
        celltype: str
            The celltype to take the cross-average prediction for
        assay: str
            The assay to take the cross-average prediction for

        Returns:
        ________
        np.array:
            The cross-average prediction 
        """
        metadata_train = self.data.metadata_train
        log_train_data = self.data.train_data
        mats = [i for i in range(metadata_train.shape[0]) if ((metadata_train["Assay Type"][i] == assay) or (metadata_train["Biosource"][i] == celltype))]
        s = None
        for i in mats :
            if s is None:
                s = log_train_data[i]
            else :
                s = s + log_train_data[i]
        s = s/len(mats)
        return(s)
    
    def get_mean_model_dictionary(self):
        """
        Provides a dictionary of the mean model calculations so that we don't have to recalculate each time we want a prediction

        Returns:
        ________
        dictionary:
            Maps celltype, assay tuples to the cross-mean average prediction. 
        """
        metadata_train = self.data.metadata_train
        metadata_valid = self.data.metadata_valid
        celltypes = metadata_train["Biosource"].tolist() + metadata_valid["Biosource"].tolist()
        assays = metadata_train["Assay Type"].tolist() + metadata_valid["Assay Type"].tolist()
        mean_model_precomp = {(celltype, assay): self.mean_model_tensor_cross(celltype, assay) for celltype, assay in zip(celltypes, assays)}
        return(mean_model_precomp)
    
    def get_mean_model_train_loss(self):
        """
        Gets the training loss from the mean model

        Returns:
        ________
        float:
            The training loss of the mean model
        """
        tse = 0
        for i in range(len(self.data.train_data)):
            celltype = self.data.metadata_train["Biosource"].iloc[i]
            assay = self.data.metadata_train["Assay Type"].iloc[i]
            pred = copy.deepcopy(self.mean_model_tensor_cross(celltype, assay)).A
            obs = copy.deepcopy(self.data.train_data[i]).A
            np.fill_diagonal(pred,0)
            np.fill_diagonal(obs, 0)
            tse += np.sum(np.square(pred.flatten() - obs.flatten())) / (pred.shape[0] ** 2 - pred.shape[0])
        return(tse / len(self.data.train_data))
    
    def get_mean_model_valid_loss2(self):
        """
        Gets the validation loss from the mean model

        Returns:
        ________
        float:
            The validation loss of the mean model
        """
        tse = 0
        for i in range(len(self.data.valid_data)):
            celltype = self.data.metadata_valid["Biosource"].iloc[i]
            assay = self.data.metadata_valid["Assay Type"].iloc[i]
            pred = copy.deepcopy(self.mean_model_tensor_cross(celltype, assay)).A
            obs = copy.deepcopy(self.data.valid_data[i]).A
            np.fill_diagonal(pred,0)
            np.fill_diagonal(obs, 0)
            tse += np.sum(np.square(pred.flatten() - obs.flatten())) / (pred.shape[0] ** 2 - pred.shape[0])
        return(tse / len(self.data.valid_data))
    
    def get_mean_model_valid_loss(self, fixed_idxs) :
        idx1 = copy.deepcopy(fixed_idxs[0])
        idx2 = copy.deepcopy(fixed_idxs[1])
        tse = 0
        if max(idx1) < 0.9 * self.data.hic_dim or max(idx1) > self.data.hic_dim:
            warnings.warn("The validation indices may be for a different chromosome")
        for i in range(len(self.data.valid_data)):
            celltype = self.data.metadata_valid["Biosource"].iloc[i]
            assay = self.data.metadata_valid["Assay Type"].iloc[i]
            pred = self.mean_model_tensor_cross(celltype, assay)
            pred_sub = pred[idx1, idx2]
            obs = self.data.valid_data[i]
            obs_sub = obs[idx1, idx2]
            tse += np.mean(np.square(pred_sub.A.flatten() - obs_sub.A.flatten()))
        return(tse / len(self.data.valid_data))
    
    def mean_same_celltype(self, celltype, assay):
        """
        Finds the celltype average prediction given a celltype and an assay (i.e. when the celltype is the same)

        Parameters:
        ___________
        celltype: str
            The celltype to take the celltype-average prediction for
        assay: str
            The assay to take the celltype-average prediction for

        Returns:
        ________
        np.array:
            The celltype-average prediction 
        """
        metadata_train = self.data.metadata_train
        log_train_data = self.data.train_data
        mats = [i for i in range(metadata_train.shape[0]) if metadata_train["Biosource"][i] == celltype]
        s = None
        for i in mats :
            if s is None:
                s = log_train_data[i]
            else :
                s = s + log_train_data[i]
        s = s/len(mats)
        return(s)
    
    def mean_same_assay(self, celltype, assay): 
        """
        Finds the assay average prediction given a celltype and an assay (i.e. when the assay is the same)

        Parameters:
        ___________
        celltype: str
            The celltype to take the assay-average prediction for
        assay: str
            The assay to take the assay-average prediction for

        Returns:
        ________
        np.array:
            The assay-average prediction 
        """
        metadata_train = self.data.metadata_train
        log_train_data = self.data.train_data
        mats = [i for i in range(metadata_train.shape[0]) if metadata_train["Assay Type"][i] == assay]
        s = None
        for i in mats :
            if s is None:
                s = log_train_data[i]
            else :
                s = s + log_train_data[i]
        s = s/len(mats)
        return(s)
    
    def get_celltype_valid_loss(self, fixed_idxs):
        """
        Validation loss for the celltype-mean baseline model
        """
        idx1 = copy.deepcopy(fixed_idxs[0])
        idx2 = copy.deepcopy(fixed_idxs[1])
        tse = 0
        if max(idx1) < 0.9 * self.data.hic_dim or max(idx1) > self.data.hic_dim:
            warnings.warn("The validation indices may be for a different chromosome")
        for i in range(len(self.data.valid_data)):
            celltype = self.data.metadata_valid["Biosource"].iloc[i]
            assay = self.data.metadata_valid["Assay Type"].iloc[i]
            pred = self.mean_same_celltype(celltype, assay)
            pred_sub = pred[idx1, idx2]
            obs = self.data.valid_data[i]
            obs_sub = obs[idx1, idx2]
            tse += np.mean(np.square(pred_sub.A.flatten() - obs_sub.A.flatten()))
        return(tse / len(self.data.valid_data))
    
    def get_assay_valid_loss(self, fixed_idxs):
        """
        Validation loss for the assay-mean baseline model
        """
        idx1 = copy.deepcopy(fixed_idxs[0])
        idx2 = copy.deepcopy(fixed_idxs[1])
        tse = 0
        if max(idx1) < 0.9 * self.data.hic_dim or max(idx1) > self.data.hic_dim:
            warnings.warn("The validation indices may be for a different chromosome")
        for i in range(len(self.data.valid_data)):
            celltype = self.data.metadata_valid["Biosource"].iloc[i]
            assay = self.data.metadata_valid["Assay Type"].iloc[i]
            pred = self.mean_same_assay(celltype, assay)
            pred_sub = pred[idx1, idx2]
            obs = self.data.valid_data[i]
            obs_sub = obs[idx1, idx2]
            tse += np.mean(np.square(pred_sub.A.flatten() - obs_sub.A.flatten()))
        return(tse / len(self.data.valid_data))
    
class DeepMatrixFactorization(torch.nn.Module):
    """
    Class to run the imputation model

    Attributes:
    ___________
    mean_model_precomp: dictionary
        Dictionary that takes (celltype, assay) pairs as key and returns a prediction from the mean model. This can be created using MeanModel.get_mean_model_dictionary()
    data: hicData
        A hicData object containing data to train and validation on.
    n_celltype_factor: int
        Number of celltype factors for the model
    n_assay_factor: int
        Number of assay factors
    n_position_factor: int
        Number of position factors
    n_distance_factor: int
        Number of distance factors
    n_node: int
        Number of hidden nodes
    n_layer: int
        Number of hidden layers
    device: str
        Name of device for pytorch (i.e. a gpu)
    residual: bool
        If true: takes the residual of the mean model for training
        If false: trains on the raw data
    debug: bool
        If true: produces more debugging messages
    droupout: float
        Percentage of dropout for the model

    """
    def __init__(self, mean_model_precomp, data, n_celltype_factor=3,
             n_assay_factor=3, n_position_factor=3, n_distance_factor=3,
             n_node=5, n_layer=2, device=None, residual=False, debug=False,
             dropout=None) :
        super().__init__() 
        n_celltype = len(data.celltype_dict)
        n_assay = len(data.assaytype_dict)
        n_positions = data.train_data[0].shape[0]
        self.n_celltype = torch.nn.Parameter(torch.tensor(n_celltype), requires_grad=False)
        self.n_assay = torch.nn.Parameter(torch.tensor(n_assay), requires_grad=False)
        self.n_position = torch.nn.Parameter(torch.tensor(n_positions), requires_grad=False)
        self.data = data
        self.celltype_factors = torch.nn.Embedding(n_celltype, n_celltype_factor)
        self.assay_factors = torch.nn.Embedding(n_assay, n_assay_factor)
        self.position_factors = torch.nn.Embedding(n_positions, n_position_factor)
        self.distance_factors = torch.nn.Embedding(n_positions, n_distance_factor)
        self.n_layer = torch.nn.Parameter(torch.tensor(n_layer), requires_grad=False)
        if device is None:
            torch.device("cpu")
        else: 
            self.device = device
        self.residual = residual
        self.debug = debug
        
        #Create initial fully connected layer taking concatenated factors to n_node
        #Then add n_position extra dimension for one-hot distance between positions
        self.fc1 = nn.Linear(n_celltype_factor + n_assay_factor +
                             2 * n_position_factor + n_distance_factor,
                             n_node)
        
        #Create n_layer more fully connected layers
        self.linears = nn.ModuleList([nn.Linear(n_node, n_node)])
        self.linears.extend([nn.Linear(n_node, n_node) for i in range(n_layer-1)])
        
        #Create final layer to output to a single value
        self.fc2 = nn.Linear(n_node, 1)
        self.device = device
        self.mean_model_precomp = mean_model_precomp
        if dropout is not None: 
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
    """
    Make a forward pass prediction using the Sphinx model

    Parameters:
    ___________
    celltype_ids: torch.tensor
        A tensor of the celltypes that we want to make predictions on
    assay_ids: torch.tensor
        A tensor of the assays that we want to make predictions on
    position_id1s:
        A tensor of the position 1s that we want to make predictions on 
    position_id2s:
        A tensor of the position 2s that we want to make predictions on

    Returns:
    ________
    torch.tensor:
        Tensor of the results of the prediction
    """
    def forward(self, celltype_ids, assay_ids, position_id1s, position_id2s) :
        celltype_factor = self.celltype_factors(celltype_ids)
        assay_factor = self.assay_factors(assay_ids)
        
        position_mins = torch.minimum(position_id1s, position_id2s)
        position_maxs = torch.maximum(position_id1s, position_id2s)
        if self.debug: 
            message = "We must have position_mins < position_maxs"
            assert all(position_mins < position_maxs), message
        position_factor1 = self.position_factors(position_mins)
        position_factor2 = self.position_factors(position_maxs)
        
        # Compute the distance factors 
        position_diff = self.distance_factors(torch.abs(position_id1s - position_id2s))
        f = torch.cat((celltype_factor, assay_factor,
                       position_factor1, position_factor2, position_diff), 1)
        x = torch.nn.functional.relu(self.fc1(f))
        if self.dropout is not None:
            x = self.dropout(x)
        for i in range(1, self.n_layer) :
            x = torch.nn.functional.relu(self.linears[i](x))
            if self.dropout is not None:
                x = self.dropout(x)
            
        #Jacob said he had trouble when he had relu on last layer
        x = self.fc2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return(x.squeeze())
    
    """
    Function for fitting the model to data

    Parameters:
        optimizer: torch.Optimizer
            an optimizer from torch.optim
        cuda: None
            legacy parameter to choose the cuda device. This is now done in the initialization of the model.
        max_epochs: int
            Number of epochs to train for
        verbose: bool
            If true, then increase the verbosity level of the model
        batchsize: int
            number of examples to include in each batch
        save_intermediate_name: str
            name of output file to export the best model weights at the end of each epoch
        valid_idxs_fn: str
            A pickle file with the indexes where [0] is the first index, and [1] is the second index to evaluate the validation set
    """
    def fit(self, optimizer, cuda, max_epochs=1000, verbose=True, batchsize=1000, save_intermediate_name="intermediate", valid_idxs_fn = None):
        train_dataset = hicDataset(self.data, "train", batchsize=batchsize, residual=self.residual)
        train_loader = DataLoader(train_dataset, batch_size=1,
                        shuffle=False, num_workers=0, pin_memory=True)
        
        # Variables to keep track of the losses
        train_loss = np.zeros(max_epochs * len(train_dataset))
        valid_loss = np.zeros(max_epochs + 1)
        valid_loss_batches = np.zeros(max_epochs + 1)
        no_model = np.zeros(max_epochs * len(train_dataset))
        best_valid_loss = 9999999
        best_state_dict = None
        counter = 0
        
        with torch.no_grad():
            self.eval()
            valid_loss[0] = self.get_valid_loss(valid_idxs_fn)
            pickle.dump([train_loss, valid_loss], open("loss.pickle", "wb"))
            if valid_loss[0] < best_valid_loss:
                best_valid_loss = valid_loss[0]
                best_state_dict = self.state_dict()
                with open(f"{save_intermediate_name}", "wb") as f:
                    torch.save(best_state_dict, f)
            valid_loss_batches[0] = counter
        
        for epoch in range(max_epochs):
            start = time.time()
            self.train()
            for x, y in train_loader: 
                x = x.squeeze().to(self.device)
                y = y.squeeze().to(self.device)
                optimizer.zero_grad()
                celltypes = x[:, 0]
                assays = x[:, 1]
                idx1s = x[:, 2]
                idx2s = x[:, 3]
                y_pred = self(celltypes, assays, idx1s, idx2s)
                loss = torch.nn.MSELoss()(y, y_pred)
                loss.backward()
                optimizer.step()
                no_model[counter] = torch.mean(torch.square(y))
                train_loss[counter] = loss
                counter += 1
            with torch.no_grad():
                self.eval()
                valid_loss[epoch + 1] = self.get_valid_loss(valid_idxs_fn)
                pickle.dump([train_loss, valid_loss], open("loss.pickle", "wb"))
                if valid_loss[epoch + 1] < best_valid_loss:
                    best_valid_loss = valid_loss[epoch + 1]
                    best_state_dict = self.state_dict()
                    with open(f"{save_intermediate_name}", "wb") as f:
                        torch.save(best_state_dict, f)
                valid_loss_batches[epoch + 1] = counter
            print(f"Epoch {epoch} took {time.time() - start} seconds, valid_loss: {valid_loss[epoch]}, train_loss: {train_loss[counter - 1]}")
        return({"train_loss": train_loss,
                "valid_loss":valid_loss,
                "valid_loss_batches": valid_loss_batches,
               "no_model": no_model})
    
    """
    Get the validation loss from the model

    Parameters:
    ___________
    valid_idxs_fn: str
        String containing the path to a pickle file that contains the indexes to evaluate the validation set on

    Returns:
    ________
    float:
        The MSE of the model on valid_idxs_fn
    """
    def get_valid_loss(self, valid_idxs_fn) :
        if valid_idxs_fn is None:
            return -1
        valid_idxs = pickle.load(open(valid_idxs_fn, "rb"))
        valid_dataset = hicDataset(self.data, "valid", fixed_idxs=valid_idxs, residual=self.residual)
        valid_loader = DataLoader(valid_dataset, batch_size=1,
                        shuffle=False, num_workers=0, pin_memory=True)
        total_loss = 0
        for x, y in valid_loader: 
                x = x.squeeze().to(self.device)
                y = y.squeeze().to(self.device)
                celltypes = x[:, 0]
                assays = x[:, 1]
                idx1s = x[:, 2]
                idx2s = x[:, 3]
                y_pred = self(celltypes, assays, idx1s, idx2s)
                loss = torch.nn.MSELoss()(y, y_pred)
                total_loss += loss
        return(total_loss / len(valid_loader))
    
    """
    Make a prediction for an entire matrix using the Sphinx model

    Parameters:
    ___________
    celltype_name: str
        The name of the celltype to make the prediction for
    assay_name: str
        The name of the assay to make the prediction for
    """
    def plot_matrix(self, celltype_name, assay_name):
        chrom = torch.tensor(0, device=self.device)
        idx1, idx2 = torch.meshgrid(
                        torch.arange(self.n_position),
                        torch.arange(self.n_position))
        idx1 = idx1.ravel().to(self.device)
        idx2 = idx2.ravel().to(self.device)
        celltype = torch.tensor(self.data.celltype_dict[celltype_name],).repeat(len(idx1)).to(self.device)
        assay = torch.tensor(self.data.assaytype_dict[assay_name]).repeat(len(idx1)).to(self.device)
        mat = self(celltype, assay, idx1, idx2).reshape(self.n_position, self.n_position)
        return(mat)
