import shap
import os
import argparse
import torch
import numpy as np
import seaborn as sns; sns.set()
import pickle as pkl
import time
import pathlib

from matplotlib import rc, rcParams
rc('font', weight='bold')
from matplotlib import rc, rcParams
rc('font', weight='bold')

from ..TSX.utils import load_simulated_data, train_model_rt, compute_median_rank, train_model_rt_binary, \
    train_model_multiclass, train_model, load_data, mem_report
from ..TSX.models import StateClassifier, RETAIN, EncoderRNN, ConvClassifier, StateClassifierMIMIC

from ..TSX.generator import JointFeatureGenerator, JointDistributionGenerator
from ..TSX.explainers import RETAINexplainer, FITExplainer, IGExplainer, FFCExplainer, \
    DeepLiftExplainer, GradientShapExplainer, AFOExplainer, FOExplainer, SHAPExplainer, \
    LIMExplainer, CarryForwardExplainer, MeanImpExplainer, TSRExplainer, GradExplainer, MockExplainer, WFITExplainer, IFITExplainer
from sklearn import metrics
from TSR.Scripts.Plotting.plot import plotExampleBox
from xgboost_model import XGBPytorchStub, remove_and_retrain
from utils import imp_ft_within_ts, plot_calibration_curve_from_pytorch

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone',
                     'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
intervention_list_plot = ['niv-vent', 'vent', 'vaso','other']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
                     'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP' , 'DiasBP' , 'MeanBP' ,
                     'RespRate' , 'SpO2' , 'Glucose','Temp']

color_map = ['#7b85d4','#f37738', '#83c995', '#d7369e','#859795', '#ad5b50', '#7e1e9c', '#0343df', '#033500', '#E0FF66', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#993F00', '#990000', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99','#003380']

ks = {'simulation_spike': 1, 'simulation': 3, 'simulation_l2x': 4}

def get_data_attributes(data_dir, data, batch_size, delay, explainer_name, activation):
    timeseries_feature_size = None
    task = None
    feature_size = None
    n_classes = 2
    
    if data == 'simulation':
        data_path = data_dir+'simulated_data'
        data_type='state'
        n_classes = 2
        feature_size = 3

    elif data == 'simulation_l2x':
        data_path = data_dir+'simulated_data_l2x'
        data_type='state'
        n_classes = 2
        feature_size = 3

        
    elif data == 'simulation_spike':
        feature_size = 3
        data_path = data_dir+'simulated_spike_data'
        data_type='spike'
        n_classes = 2 # use with state-classifier
        
        if explainer_name=='retain':
            activation = torch.nn.Softmax()
        else:
            activation = torch.nn.Sigmoid()
        
        if batch_size != 100:
            batch_size = 200

        if delay != 0:
            assert delay > 0
            data_path = f'{data_dir}simulated_spike_data_delay_{delay}'

    elif data == 'mimic':
        data_type = 'mimic'
        n_classes = 2
        
        timeseries_feature_size = len(feature_map_mimic)
        task = 'mortality'
        
    elif data == 'mimic_int':
        data_type = 'real'
        n_classes = 4
        
        #change this to softmax for suresh et al
        activation = torch.nn.Sigmoid()
        #activation = torch.nn.Softmax(-1)

        if batch_size != 100:
            batch_size = 256
            
        timeseries_feature_size = len(feature_map_mimic)
        task = 'intervention'
       
    
    return feature_size, data_path, data_type, n_classes, activation, batch_size, timeseries_feature_size, task

def run_baseline(explainer_name='fit', 
                 data='simulation',
                 train=True, 
                 train_gen=True, 
                 binary=False,
                 generator_type='history', 
                 out_path='./output/',
                 data_dir='./data/',
                 mimic_path='', 
                 xgb=False,
                 roar=False,
                 skip=False,
                 skip_explanation=False,
                 gt='true_model',
                 cv=0, 
                 N=1,
                 delay=0,
                 gen_epochs=300,
                 batch_size=100,
                 samples=10):
    
    np.random.seed(cv)
    torch.manual_seed(cv)
    model = None
    generator = None
    explainer = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
        
    output_path = out_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    plot_path = os.path.join('./plots/%s' % data)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
        
    # XGB Params
    xgb_window_size = 10
    xgb_buffer_size = 0
    xgb_target_size = 1
    
    # Activation
    activation = torch.nn.Softmax(-1)

    # Get Data Attributes
    feature_size, data_path, data_type, n_classes, activation, batch_size, timeseries_feature_size, task = get_data_attributes(data_dir, data, batch_size, delay, explainer_name, activation)
    
    if delay != 0:
        assert delay > 0
        data = f'simulation_spike_delay_{delay}'
        
    # Load Data
    if data == 'mimic' or data=='mimic_int':
        if mimic_path is None:
            raise ValueError('Specify the data directory containing processed mimic data')
        p_data, train_loader, valid_loader, test_loader = load_data(batch_size=batch_size, \
            path=mimic_path,task=task,cv=cv ,test_bs=batch_size)
        feature_size = p_data.feature_size
        class_weight = p_data.pos_weight
    else:
        _, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=batch_size, datapath=data_path,
                                                                         percentage=0.8, data_type=data_type,cv=cv)
        
    print(f"Using dataset: {data_dir}, {data}, {data_path}")
    print(f"Features: {feature_size}    Classes: {n_classes}")
    print(f"Batch size: {batch_size}     Activation: {activation}")
        
    # Deal with any conflicting arguments
    assert xgb or not roar
    
    # Setup Remove And Retrain
    if roar and skip:
        importance_scores = None
        importance_path = os.path.join(output_path + f'/{data}', '%s_test_importance_scores_%d.pkl' % (explainer_name, cv))
        if os.path.exists(importance_path):
            with open(importance_path, 'rb') as imp_file:
                importance_scores = pkl.load(imp_file)
        remove_and_retrain(train_loader, test_loader, xgb_window_size, xgb_buffer_size, xgb_target_size,
                           f'plots/{data}/xgb_roar_{explainer_name}', explainer_name, importance_scores)
        quit()

    # === MODEL ===
    
    # === RETAIN ===
    if explainer_name == 'retain':
        if data=='mimic' or data=='simulation' or data=='simulation_l2x':
            model = RETAIN(dim_input=feature_size, dim_emb=128, dropout_emb=0.4, dim_alpha=8, dim_beta=8,
                       dropout_context=0.4, dim_output=2)
        elif data=='mimic_int':
            model = RETAIN(dim_input=feature_size, dim_emb=32, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=n_classes)
        elif data.startswith('simulation_spike'):
            model = RETAIN(dim_input=feature_size, dim_emb=4, dropout_emb=0.4, dim_alpha=16, dim_beta=16,
                       dropout_context=0.4, dim_output=n_classes)
        explainer = RETAINexplainer(model, data)
        if train:
            t0 = time.time()
            if data=='mimic' or data=='simulation' or data=='simulation_l2x':
                explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-3, plot=True, epochs=50)
            else:
                explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-4, plot=True, epochs=100,cv=cv)
            print('Total time required to train retain: ', time.time() - t0)
        else:
            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (data, 'retain', cv))))
            
    else:
        # === XGB ===
        if xgb:
            model = XGBPytorchStub(train_loader, valid_loader, xgb_window_size, xgb_buffer_size, xgb_target_size,
                                   os.path.join('./ckpt/%s/%s_%d.model' % (data, 'xgb_model',cv)), True)
        # === RNN ===
        else:
            if not binary:
                if data=='mimic_int':
                    model = StateClassifierMIMIC(feature_size=feature_size, n_state=n_classes, hidden_size=128, rnn='LSTM')
                else:
                    model = StateClassifier(feature_size=feature_size, n_state=n_classes, hidden_size=200, rnn='GRU')
            else:
                model = EncoderRNN(feature_size=feature_size, hidden_size=10, regres=True, return_all=False, data=data, rnn="GRU")

            
            if train:
                if not binary:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
                    if data=='mimic':
                        train_model(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=100,
                                    device=device, experiment='model',cv=cv)

                    elif 'simulation' in data:
                        train_model_rt(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, n_epochs=30,
                                   device=device, experiment='model', data=data, cv=cv)

                    elif data=='mimic_int':
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
                        if type(activation).__name__==type(torch.nn.Softmax(-1)).__name__: #suresh et al
                            train_model_multiclass(model=model, train_loader=train_loader, valid_loader=test_loader,
                            optimizer=optimizer, n_epochs=50, device=device, experiment='model', data=data,num=5,
                            loss_criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).to(device)),cv=cv)
                        else:
                            train_model_multiclass(model=model, train_loader=train_loader, valid_loader=test_loader,
                            optimizer=optimizer, n_epochs=25, device=device, experiment='model', data=data, num=5,
                            #loss_criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight).cuda()),cv=cv)
                            loss_criterion=torch.nn.BCEWithLogitsLoss(),cv=cv)
                            #loss_criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).cuda()),cv=cv)
                            #loss_criterion=torch.nn.CrossEntropyLoss(),cv=cv)
                else:
                    #this learning rate works much better for spike data
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                    if data=='mimic':
                        train_model(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=200,
                                    device=device, experiment='model',cv=cv)
                    else:
                        train_model_rt_binary(model, train_loader, valid_loader, optimizer=optimizer, n_epochs=250,
                                   device=device, experiment='model', data=data, cv=cv)

            model.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (data, 'model', cv))))
            
            if skip_explanation:
                if model:
                    del model
                if explainer:
                    del explainer
                if generator:
                    del generator
                torch.cuda.empty_cache()
                return 

        # === EXPLANATION ===
        if N > 1 or explainer_name == 'ifit':
            assert explainer_name in ['fit', 'ifit']
            inverse = explainer_name == 'ifit'
            explainer = WFITExplainer(model, N, inverse, train_loader, test_loader, data + f'_{N}',
                                      activation=None if xgb else torch.nn.Softmax(-1), train_generators=train_gen, n_samples=samples, cv=cv)

        elif explainer_name == 'fit':
            if generator_type=='history':
                generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=data)
                
                if xgb:
                    if data != 'mimic_int' and not data.startswith('simulation_spike'):
                        n_classes = 2
                    explainer = FITExplainer(model, activation=lambda x: x, n_classes=n_classes, n_samples=samples)
                elif data=='mimic_int' or data.startswith('simulation_spike'):
                    explainer = FITExplainer(model, activation=torch.nn.Sigmoid(), n_classes=n_classes, n_samples=samples)
                else:
                    explainer = FITExplainer(model, n_samples=samples)

                if train_gen:
                    explainer.fit_generator(generator, train_loader, valid_loader, cv=cv, n_epochs=gen_epochs)
                else:
                    generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s_%d.pt' % (data, 'joint_generator',cv))))
                    generator.to(device)
                    explainer.generator = generator
                
            elif generator_type=='no_history':
                generator = JointDistributionGenerator(n_components=5, train_loader=train_loader)
                if data=='mimic_int' or data.startswith('simulation_spike'):
                    explainer = FITExplainer(model, generator, activation=torch.nn.Sigmoid(), n_samples=samples)
                else:
                    explainer = FITExplainer(model, generator, n_samples=samples)

        elif explainer_name == 'integrated_gradient':
            if data=='mimic_int' or data.startswith('simulation_spike'):
                explainer = IGExplainer(model,activation=activation)
            else:
                explainer = IGExplainer(model)

        elif explainer_name == 'deep_lift':
            if data=='mimic_int' or data.startswith('simulation_spike'):
                explainer = DeepLiftExplainer(model, activation=activation)
            else:
                explainer = DeepLiftExplainer(model)

        elif explainer_name == 'fo':
            if data=='mimic_int' or data.startswith('simulation_spike'):
                explainer = FOExplainer(model,activation=activation)
            else:
                explainer = FOExplainer(model)

        elif explainer_name == 'afo':
            if data=='mimic_int' or data.startswith('simulation_spike'):
                explainer = AFOExplainer(model, train_loader,activation=activation)
            else:
                explainer = AFOExplainer(model, train_loader)

        elif explainer_name == 'carry_forward':
            explainer = CarryForwardExplainer(model, train_loader)

        elif explainer_name == 'mean_imp':
            explainer = MeanImpExplainer(model, train_loader)

        elif explainer_name == 'gradient_shap':
            if data=='mimic_int' or data.startswith('simulation_spike'):
                explainer = GradientShapExplainer(model, activation=activation)
            else:
                explainer = GradientShapExplainer(model)

        elif explainer_name == 'ffc':
            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=data)
            if train:
                if data=='mimic_int' or data.startswith('simulation_spike'):
                    explainer = FFCExplainer(model,activation=activation)
                else:
                    explainer = FFCExplainer(model)
                explainer.fit_generator(generator, train_loader, valid_loader)
            else:
                generator.load_state_dict(torch.load(os.path.join('./ckpt/%s/%s.pt' % (data, 'joint_generator'))))
                if data=='mimic_int' or data.startswith('simulation_spike'):
                    explainer = FFCExplainer(model, generator, activation=activation)
                else:
                    explainer = FFCExplainer(model, generator)

        elif explainer_name == 'shap':
            explainer = SHAPExplainer(model, train_loader)

        elif explainer_name == 'lime':
            if data=='mimic_int' or data.startswith('simulation_spike'):
                explainer = LIMExplainer(model, train_loader, activation=activation,n_classes=n_classes)
            else:
                explainer = LIMExplainer(model, train_loader)

        elif explainer_name == 'retain':
             explainer = RETAINexplainer(model,self.data)

        elif explainer == 'grad':
            explainer = GradExplainer(model)

        elif explainer_name == 'grad_tsr':
            explainer = TSRExplainer(model, "Grad")
        elif explainer_name == 'ig_tsr':
            explainer = TSRExplainer(model, "IG")

        elif explainer_name == 'ifit':
            explainer = IFITExplainer(model, activation=None if xgb else torch.nn.Softmax(-1))

        elif explainer_name == 'mock':
            explainer = MockExplainer()

        else:
            raise ValueError('%s explainer not defined!' % explainer)
            

    # === EVALUATION ===
    
    # Load ground truth for simulations
    if data_type == 'state':
        with open(os.path.join(data_path, 'state_dataset_importance_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_states_test.pkl'), 'rb') as f:
            state_test = pkl.load(f)
        with open(os.path.join(data_path, 'state_dataset_logits_test.pkl'), 'rb') as f:
            logits_test = pkl.load(f)
    elif data_type == 'spike':
        with open(os.path.join(data_path, 'gt_test.pkl'), 'rb') as f:
            gt_importance_test = pkl.load(f)

    importance_scores = []
    ranked_feats=[]
    xs = []
    ys = []
    n_samples = 1
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)

        #model.train()
        model.to(device)
        x = x.to(device)
        y = y.to(device)

        score = explainer.attribute(x, y if data=='mimic' else y[:, -1].long())

        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0) + 1) \
                                    for n in range(x.shape[0])])
        importance_scores.append(score)
        ranked_feats.append(ranked_features)

    calibration_path = f'plots/calibration/{data}/{"xgb" if xgb else "rnn"}_model_{cv}.png'
    plot_calibration_curve_from_pytorch(model, test_loader, calibration_path, activation)

    importance_scores = np.concatenate(importance_scores, 0)
    xs = np.concatenate(xs, 0)
    ys = np.concatenate(ys, 0)
    pathlib.Path(output_path + f'/{data}').mkdir(parents=True, exist_ok=True)
    print('Saving file to ', os.path.join(output_path + f'/{data}', '%s_test_importance_scores_%d.pkl' % (explainer_name, cv)))
    with open(os.path.join(output_path + f'/{data}', '%s_test_importance_scores_%d.pkl' % (explainer_name, cv)), 'wb') as f:
        pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

    if roar:
        remove_and_retrain(train_loader, test_loader, xgb_window_size, xgb_buffer_size, xgb_target_size,
                           f'plots/{data}/xgb_roar_{explainer_name}', explainer_name, importance_scores)

    ranked_feats = np.concatenate(ranked_feats,0)
    with open(os.path.join(output_path, '%s_test_ranked_scores.pkl' % explainer_name), 'wb') as f:
        pkl.dump(ranked_feats, f, protocol=pkl.HIGHEST_PROTOCOL)

    if 'simulation' in data:
        gt_soft_score = np.zeros(gt_importance_test.shape)
        gt_importance_test.astype(int)

        for i in range(20):
            if np.sum(gt_importance_test[i]) == 0: 
                continue
            plotExampleBox(np.abs(importance_scores[i]), f'plots/{data}/{explainer_name}_attributions_{i}', greyScale=True)
            plotExampleBox(gt_importance_test[i], f'plots/{data}/ground_truth_attributions_{i}', greyScale=True)
            plotExampleBox(xs[i], f'plots/{data}/data_{i}', greyScale=True)
            plotExampleBox(np.array([ys[i]]), f'plots/{data}/labels_{i}', greyScale=True)

        imp_ft_within_ts(importance_scores, gt_importance_test)

        gt_score = gt_importance_test.flatten()
        explainer_score = importance_scores.flatten()
        if explainer_name=='deep_lift' or explainer_name=='integrated_gradient' or explainer=='gradient_shap':
            explainer_score = np.abs(explainer_score)
        # TODO: Why do we get NaNs?
        explainer_score = np.nan_to_num(explainer_score)
        auc_score = metrics.roc_auc_score(gt_score, explainer_score)
        aupr_score = metrics.average_precision_score(gt_score, explainer_score)

        _, median_rank, _= compute_median_rank(ranked_feats, gt_soft_score, soft=True,K=4)
        print('auc:', auc_score, ' aupr:', aupr_score)
        
        if model:
            del model
        if explainer:
            del explainer
        if generator:
            del generator
        torch.cuda.empty_cache()
        #mem_report()
        return auc_score, aupr_score

        
def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    parser.add_argument('--explainer_name', type=str, default='fit', help='Explainer model')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_gen', action='store_true')
    parser.add_argument('--generator_type', type=str, default='history')
    parser.add_argument('--out_path', type=str, default='./output/')
    parser.add_argument('--mimic_path', type=str)
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--xgb', action='store_true', default=False)
    parser.add_argument('--roar', action='store_true', default=False)
    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--skip_explanation', action='store_true', default=False, help='Return immediately after training model')
    parser.add_argument('--gt', type=str, default='true_model', help='specify ground truth score')
    parser.add_argument('--cv', type=int, default=0, help='cross validation')
    parser.add_argument('--N', type=int, default=1, help='fit/ifit window size')
    parser.add_argument('--samples', type=int, default=10, help='number of samples to be taken from generator for fit/ifit')
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--gen_epochs', type=int, default=300, help='number of epochs to train the generator')
    args = parser.parse_args() # from sys.argv
    return args

if __name__ == '__main__':
    args = parse_args()
    run_baseline(**vars(args))
