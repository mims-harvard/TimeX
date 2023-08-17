from collections import defaultdict
import torch
import torch.nn.functional as F
import argparse
import numpy as np

from txai.vis.visualize_mv6 import vis_concepts, visualize_explanations, visualize_explanations_new

# Models:
from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.bc_model import BCExplainModel

from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.predictors.select_models import cosine_sim 
from txai.utils.cl_metrics import sim_mat
from txai.utils.data.preprocess import process_Epilepsy, process_MITECG
from txai.models.run_model_utils import batch_forwards

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--embedding', action = 'store_true')
    parser.add_argument('--kmeans_proto', action = 'store_true')
    parser.add_argument('--random_proto', action = 'store_true')
    parser.add_argument('--class_num', type=int, default = 0)
    parser.add_argument('--split_no', type=int, default = 1)
    parser.add_argument('--discrete', action='store_true', help = 'Shows mask as discrete object')
    parser.add_argument('--sample_seed', type = int)
    parser.add_argument('--save_emb_name', type = str)
    parser.add_argument('--topk', type = int, default = None)
    parser.add_argument('--show_concepts', action = 'store_true', help = 'shows discovered concepts, if applicable')
    parser.add_argument('--savepdf', type = str, default = None)

    parser.add_argument('--org_v', action = 'store_true')

    args = parser.parse_args()

    metrics = defaultdict(list)

    np.random.seed(0)
    torch.manual_seed(0)
    for split in range(5):
        args.split_no = split + 1
        model_path = args.model_path.replace("split=1", f"split={split + 1}")

        D = args.dataset.lower()

        if D == 'freqshape':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShape')
            test = D['test']
            train = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y)
        elif D == 'seqcombsingle':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingle')
            test = D['test']
        elif D == 'scs_better':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
            test = D['test']
        elif D == 'freqshapeud':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/FreqShapeUD')
            test = D['test']
        elif D == 'scs_fixone':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleFixOne')
            test = D['test']
        elif D == 'epilepsy':
            _, _, test = process_Epilepsy(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/Epilepsy/')
            test = (test.X, test.time, test.y)
        elif D == 'mitecg_simple':
            _, _, test = process_MITECG(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Simple/')
            test = (test.X, test.time, test.y)
        elif D == 'lowvardetect':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/LowVarDetect')
            test = D['test']
        elif D == 'seqcomb_mv':
            D = process_Synth(split_no = args.split_no, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombMV')
            test = D['test']
        elif D == 'mitecg_hard':
            train, _, test, _ = process_MITECG(split_no = args.split_no, hard_split = True, need_binarize = True, device = device, base_path = 'datasets/drive/datasets_and_models/MITECG-Hard/')
            # train, _, test, _ = process_MITECG(split_no = args.split_no, hard_split = True, need_binarize = True, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG-Hard/')
            test = (test.X, test.time, test.y)

        # Loading:
        print('Loading model at {}'.format(model_path))
        sdict, config = torch.load(model_path)
        #print('Config:\n', config)
        #exit()
        if args.org_v:
            model = Modelv6_v2(**config)
        else:
            model = BCExplainModel(**config)
        model.load_state_dict(sdict)
        model.eval()
        model.to(device)

        if args.kmeans_proto:
            # replace model.prototypes with kmeans centroids
            from sklearn.cluster import KMeans

            X_train, times_train, *_ = train.get_all()
            with torch.no_grad():
                out_train = batch_forwards(model, X_train, times_train, batch_size=32, org_v=args.org_v)
            ztrain = out_train['z_mask_list'].squeeze(-1)
            kmeans = KMeans(n_clusters=model.n_prototypes)
            kmeans.fit(ztrain)
            model.prototypes = torch.nn.Parameter(torch.from_numpy(kmeans.cluster_centers_).float().to(model.prototypes.device))

        if args.random_proto:
            # replace prototypes with random 
            model.prototypes = torch.nn.Parameter(torch.randn_like(model.prototypes))
        
        # Experiment, evaluate clustering of prototypes
        from sklearn.linear_model import LogisticRegression    
        from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import f1_score

        with torch.no_grad():

            # normalized prototypes
            proto_norm = F.normalize(model.prototypes, dim = 1)


            # inference for z vectors on train
            X_train, times_train, y_train, _ = train.get_all()
            out_train = batch_forwards(model, X_train, times_train, batch_size=32, org_v=args.org_v)
            ztrain = out_train['z_mask_list'].squeeze(-1).to(device)

            # cosine similarities between z_mask and prototypes
            ztrain_norm = F.normalize(ztrain, dim = 1)
            sim_train = torch.matmul(ztrain_norm, proto_norm.transpose(0, 1))

            # find k most similar vectors and use those
            # best_k_indices = sim_train.sum(0).topk(5).indices
            
            # use all k
            best_k_indices = torch.arange(sim_train.shape[1], device=sim_train.device)

            knn = KNeighborsClassifier(n_neighbors=1).fit(model.prototypes[best_k_indices, :].cpu().numpy(), np.arange(len(best_k_indices)))

            # train lr on similarity vectors
            logistic_regression = LogisticRegression(max_iter=10_000).fit(sim_train[:, best_k_indices].cpu().numpy(), y_train.cpu().numpy())

            # inference for z vectors on test
            X_test, times_test, y_test = test
            out_test = batch_forwards(model, X_test, times_test, batch_size=32, org_v=args.org_v)
            ztest = out_test['z_mask_list'].squeeze(-1).to(device)

            # cosine similarities between z_mask and prototypes
            ztest_norm = F.normalize(ztest, dim = 1)
            sim_test = torch.matmul(ztest_norm, proto_norm.transpose(0, 1))

            # generate predictions
            y_pred = logistic_regression.predict(sim_test[:, best_k_indices].cpu().numpy())
            knn_pred = knn.predict(ztest.cpu().numpy())
            # metrics["f1"].append(f1_score(y_test.cpu().numpy(), y_pred))
            metrics["nmi"].append(normalized_mutual_info_score(y_test.cpu().numpy(), knn_pred))
            metrics["ari"].append(adjusted_rand_score(y_test.cpu().numpy(), knn_pred))
            metrics["silhouette"].append(silhouette_score(ztest.cpu().numpy(), knn_pred))


    for name, metric in metrics.items():
        metric = np.array(metric)
        print(f"{name} {metric.mean():.3f} +- {metric.std() / np.sqrt(len(metric)): .3f}")
