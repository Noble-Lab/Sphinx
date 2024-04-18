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


class hicData():
    def __init__(self, folder, metadata_train, metadata_valid, metadata_test, resolution, chromosome, scale):
        self.folder = folder
        self.metadata_train = metadata_train
        self.metadata_valid = metadata_valid
        self.metadata_test = metadata_test
        self.resolution = resolution
        self.chromosome = chromosome
        self.scale = scale
        
        train_names = [os.path.basename(x) for x in metadata_train["ProcessedFileHref"]]
        valid_names = [os.path.basename(x) for x in metadata_valid["ProcessedFileHref"]]
        test_names = [os.path.basename(x) for x in metadata_test["ProcessedFileHref"]]
        
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
        unique_labels = np.unique(labels)
        return({label: i for i, label in enumerate(unique_labels)})
        
    def get_matrices_unpruned(self, file_names):
        coolers = [cooler.Cooler(f"{self.folder}{fn}::/resolutions/{self.resolution}") for fn in file_names]
        coolers = hicData.get_sparse_matrices(coolers, self.chromosome)
        coolers = hicData.log_plus_one_matrices(coolers)
        return(coolers)
    
    def get_matrices(self, file_names):
        coolers = [cooler.Cooler(f"{self.folder}{fn}::/resolutions/{self.resolution}") for fn in file_names]
        coolers = hicData.get_sparse_matrices(coolers, self.chromosome)
        coolers = hicData.log_plus_one_matrices(coolers)
        coolers = hicData.prune(coolers, self.whitelist)
        coolers = hicData.normalize_matrices(coolers, self.scale)
        return(coolers)
    
    def find_celltype_assay(self, celltype, assay):
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
        start = time.time()
        assert(mat.shape[0] == mat.shape[1])
        n = mat.shape[0]
        avgs = np.zeros(n)
        for i in range(n):
            avgs[i] = np.mean(mat.diagonal(i))
        return(avgs)
    
    def make_cdp(self, model, celltype, assay, ax=None, xmin=None, xmax = None, ymin=None, ymax=None):
        obs, metadata_name = self.find_celltype_assay(celltype, assay)
        mean_mat = m.mean_model_tensor_cross(celltype, assay).todense().A
        pred_resid = d.plot_matrix(celltype, assay).cpu().detach().numpy()
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
        return np.linalg.eig(obs)[1][0] #returns first eigenvector
    
    def make_eigenvector(self, model, celltype, assay, ax=None, xmin=None, xmax = None, ymin=None, ymax=None):
        obs, metadata_name = self.find_celltype_assay(celltype, assay)
        mean_mat = m.mean_model_tensor_cross(celltype, assay).todense().A
        pred_resid = d.plot_matrix(celltype, assay).cpu().detach().numpy()
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
    

"""
This class is used to create data loaders using pytorch.
"""
class hicDataset(torch.utils.data.Dataset):
    """
    Initialization function the mean model class. 
    @Param hicData: A class of hicData to use for the mean model
    @
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
    Initialization function the mean model class. 
    @Param data: A class of hicData to use for the mean model
    """
    def __init__(self, data):
        self.data = data
    
    def mean_model_tensor_cross(self, celltype, assay) :
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
        metadata_train = self.data.metadata_train
        metadata_valid = self.data.metadata_valid
        celltypes = metadata_train["Biosource"].tolist() + metadata_valid["Biosource"].tolist()
        assays = metadata_train["Assay Type"].tolist() + metadata_valid["Assay Type"].tolist()
        mean_model_precomp = {(celltype, assay): self.mean_model_tensor_cross(celltype, assay) for celltype, assay in zip(celltypes, assays)}
        return(mean_model_precomp)
    
    def get_mean_model_train_loss(self):
        # raise ValueError("I don't know any other errors to do")
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
        # raise ValueError("I don't know any other errors to do")
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
    @param mean_model_precomp: a dictionary that takes celltype and assay
        and returns a precomputed mean model
    @param data: A class of hicData to use for imputation
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
    Make a forward pass prediction 
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
    
    def plot_matrix(self, celltype_name, assay_name):
        chrom = torch.tensor(0, device=cuda0)
        idx1, idx2 = torch.meshgrid(
                        torch.arange(d.n_position),
                        torch.arange(d.n_position))
        idx1 = idx1.ravel().to(cuda0)
        idx2 = idx2.ravel().to(cuda0)
        celltype = torch.tensor(d.data.celltype_dict[celltype_name],).repeat(len(idx1)).to(cuda0)
        assay = torch.tensor(d.data.assaytype_dict[assay_name]).repeat(len(idx1)).to(cuda0)
        mat = self(celltype, assay, idx1, idx2).reshape(d.n_position, d.n_position)
        return(mat)
