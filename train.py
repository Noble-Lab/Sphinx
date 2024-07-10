from imputation import * 
import argparse

folder = "/net/noble/vol1/home/noble/proj/2019_wnoble_impute3d/results/bill/2020-12-08download/"
metadata_train = pickle.load(open("metadata_train.pickle", "rb"))
metadata_valid = pickle.load(open("metadata_valid.pickle", "rb"))
metadata_test = pickle.load(open("metadata_test.pickle", "rb"))
metadata = pickle.load(open("metadata.pickle", "rb"))
resolution = 100000
chromosome = 19
scale = 1e5
batchsize = 10000
cuda0 = torch.device("cuda:0") 
residual = True

h = hicData(folder, metadata_train, metadata_valid, metadata_test, resolution, chromosome, scale)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--assay", help="Number of assay factors", type=int, default=None)
parser.add_argument("--celltype", help="Number of celltype factors", type=int, default=None)
parser.add_argument("--position", help="Number of position factors", type=int, default=None)
parser.add_argument("--node", help="Number of nodes", type=int, default=None)
parser.add_argument("--layer", help="Number of hidden layers", type=int, default=None)
parser.add_argument("--lr", help="Learning rate", type=float, default=None)
parser.add_argument("--dropout", help="Dropout rate", type=float, default=None)
args = parser.parse_args()

n_celltype_factor = args.celltype
n_assay_factor = args.assay
n_position_factor = args.position
n_node = args.node
n_layer = args.layer
lr = args.lr
dropout = args.dropout
print(f"Arguments were n_celltype_factor: {n_celltype_factor} n_assay_factor: {n_assay_factor} n_position_factor: {n_position_factor} n_node: {n_node} n_layer: {n_layer} lr: {lr} dropout: {dropout}")
      
torch.autograd.set_detect_anomaly(False)
m = MeanModel(h)
mean_model_precomp = m.get_mean_model_dictionary()
d = DeepMatrixFactorization(mean_model_precomp, h, device=cuda0, residual=residual, n_celltype_factor=n_celltype_factor, n_assay_factor=n_assay_factor, n_position_factor=n_position_factor, n_distance_factor=n_position_factor, n_node=n_node, n_layer=n_layer, debug=False, dropout=dropout)
d.to(cuda0)
optimizer = torch.optim.Adam(d.parameters(), lr=lr)
losses = d.fit(optimizer, None, max_epochs=50, batchsize=batchsize, valid_idxs_fn="valid_idxs.pickle")
pickle.dump(d, open(f"models/model_n_celltype_factor{n_celltype_factor}n_assay_factor{n_assay_factor}n_position_factor{n_position_factor}n_distance_factor{n_position_factor}n_node{n_node}n_layer{n_layer}dropout{dropout}lr{lr:.04f}.pickle", "wb"))
pickle.dump(losses, open(f"models/losses_n_celltype_factor{n_celltype_factor}n_assay_factor{n_assay_factor}n_position_factor{n_position_factor}n_distance_factor{n_position_factor}n_node{n_node}n_layer{n_layer}dropout{dropout}lr{lr:.04f}.pickle", "wb"))
