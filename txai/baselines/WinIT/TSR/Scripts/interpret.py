import os
import logging
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation,
    Occlusion

)

from .tsr import get_tsr_saliency
from . import Helper
from .Helper import checkAccuracy
from .getSaliencyMapMetadata import getSaliencyMapMetadata
from .Plotting.plot import *
from inverse_fit import inverse_fit_attribute, wfit_attribute
from FIT.TSX.generator import JointFeatureGenerator
from FIT.TSX.explainers import FITExplainer


def run_saliency_methods(saliency_methods, pretrained_model, test_shape, train_loader, test_loader, device, 
                         model_type, model_name, saliency_dir, tsr_graph_dir=None, tsr_inputs_to_graph=()):
    _, num_timesteps, num_features = test_shape
    
    run_grad = "Grad" in saliency_methods
    run_grad_tsr = "Grad_TSR" in saliency_methods
    run_ig = "IG" in saliency_methods
    run_ig_tsr = "IG_TSR" in saliency_methods
    run_dl = "DL" in saliency_methods
    run_gs = "GS" in saliency_methods
    run_dls = "DLS" in saliency_methods
    run_dls_tsr = "DLS_TSR" in saliency_methods
    run_sg = "SG" in saliency_methods
    run_shapley_sampling = "ShapleySampling" in saliency_methods
    run_feature_permutation = "FeaturePermutation" in saliency_methods
    run_feature_ablation = "FeatureAblation" in saliency_methods
    run_occlusion = "Occlusion" in saliency_methods
    run_fit = "FIT" in saliency_methods
    run_ifit = "IFIT" in saliency_methods
    run_wfit = "WFIT" in saliency_methods
    run_iwfit = "IWFIT" in saliency_methods

    if run_grad or run_grad_tsr:
        Grad = Saliency(pretrained_model)
    if run_grad:
        rescaledGrad = np.zeros(test_shape)
    if run_grad_tsr:
        rescaledGrad_TSR = np.zeros(test_shape)

    if run_ig or run_ig_tsr:
        IG = IntegratedGradients(pretrained_model)
    if run_ig:
        rescaledIG = np.zeros(test_shape)
    if run_ig_tsr:
        rescaledIG_TSR = np.zeros(test_shape)

    if run_dl:
        rescaledDL = np.zeros(test_shape)
        DL = DeepLift(pretrained_model)

    if run_gs:
        rescaledGS = np.zeros(test_shape)
        GS = GradientShap(pretrained_model)

    if run_dls or run_dls_tsr:
        DLS = DeepLiftShap(pretrained_model)
    if run_dls:
        rescaledDLS = np.zeros(test_shape)
    if run_dls_tsr:
        rescaledDLS_TSR = np.zeros(test_shape)

    if run_sg:
        rescaledSG = np.zeros(test_shape)
        Grad_ = Saliency(pretrained_model)
        SG = NoiseTunnel(Grad_)

    if run_shapley_sampling:
        rescaledShapleySampling = np.zeros(test_shape)
        SS = ShapleyValueSampling(pretrained_model)

    if run_gs:
        rescaledFeaturePermutation = np.zeros(test_shape)
        FP = FeaturePermutation(pretrained_model)

    if run_feature_ablation:
        rescaledFeatureAblation = np.zeros(test_shape)
        FA = FeatureAblation(pretrained_model)

    if run_occlusion:
        rescaledOcclusion = np.zeros(test_shape)
        OS = Occlusion(pretrained_model)

    if run_fit:
        rescaledFIT = np.zeros(test_shape)
        FIT = FITExplainer(pretrained_model, ft_dim_last=True)
        generator = JointFeatureGenerator(num_features, data='none')
        # TODO: Increase epochs
        FIT.fit_generator(generator, train_loader, test_loader, n_epochs=300)

    if run_ifit:
        rescaledIFIT = np.zeros(test_shape)
    if run_wfit:
        rescaledWFIT = np.zeros(test_shape)
    if run_iwfit:
        rescaledIWFIT = np.zeros(test_shape)

    idx = 0
    mask = np.zeros((num_timesteps, num_features), dtype=int)
    for i in range(num_timesteps):
        mask[i, :] = i

    for i, (samples, labels) in enumerate(test_loader):
        input = samples.reshape(-1, num_timesteps, num_features).to(device)
        input = Variable(input, volatile=False, requires_grad=True)

        batch_size = input.shape[0]
        baseline_single = torch.from_numpy(np.random.random(input.shape)).to(device)
        baseline_multiple = torch.from_numpy(np.random.random((input.shape[0] * 5, input.shape[1], input.shape[2]))).to(
            device)
        inputMask = np.zeros((input.shape))
        inputMask[:, :, :] = mask
        inputMask = torch.from_numpy(inputMask).to(device)
        mask_single = torch.from_numpy(mask).to(device)
        mask_single = mask_single.reshape(1, num_timesteps, num_features).to(device)
        labels = torch.tensor(labels.int().tolist()).to(device)

        if run_grad:
            attributions = Grad.attribute(input, target=labels)
            rescaledGrad[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)
        if run_grad_tsr:
            rescaledGrad_TSR[idx:idx + batch_size, :, :] = get_tsr_saliency(Grad, input, labels,
                                                                            graph_dir=tsr_graph_dir,
                                                                            graph_name=f'{model_name}_{model_type}_Grad_TSR',
                                                                            inputs_to_graph=tsr_inputs_to_graph, cur_batch=i)

        if run_ig:
            attributions = IG.attribute(input, baselines=baseline_single, target=labels)
            rescaledIG[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)
        if run_ig_tsr:
            rescaledIG_TSR[idx:idx + batch_size, :, :] = get_tsr_saliency(IG, input, labels,
                                                                          baseline=baseline_single, graph_dir=tsr_graph_dir,
                                                                          graph_name=f'{model_name}_{model_type}_IG_TSR',
                                                                          inputs_to_graph=tsr_inputs_to_graph, cur_batch=i)

        if run_dl:
            attributions = DL.attribute(input, baselines=baseline_single, target=labels)
            rescaledDL[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_gs:
            attributions = GS.attribute(input, baselines=baseline_multiple, stdevs=0.09, target=labels)
            rescaledGS[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_dls:
            attributions = DLS.attribute(input, baselines=baseline_multiple, target=labels)
            rescaledDLS[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)
        if run_dls_tsr:
            rescaledDLS_TSR[idx:idx + batch_size, :, :] = get_tsr_saliency(DLS, input, labels,
                                                                          baseline=baseline_multiple, graph_dir=tsr_graph_dir,
                                                                          graph_name=f'{model_name}_{model_type}_DLS_TSR',
                                                                           inputs_to_graph=tsr_inputs_to_graph, cur_batch=i)

        if run_sg:
            attributions = SG.attribute(input, target=labels)
            rescaledSG[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_shapley_sampling:
            attributions = SS.attribute(input, baselines=baseline_single, target=labels, feature_mask=inputMask)
            rescaledShapleySampling[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_feature_permutation:
            attributions = FP.attribute(input, target=labels, perturbations_per_eval=input.shape[0],
                                        feature_mask=mask_single)
            rescaledFeaturePermutation[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features,
                                                                                                        attributions)

        if run_feature_ablation:
            attributions = FA.attribute(input, target=labels)
            # perturbations_per_eval= input.shape[0],\
            # feature_mask=mask_single)
            rescaledFeatureAblation[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_occlusion:
            attributions = OS.attribute(input, sliding_window_shapes=(1, num_features), target=labels,
                                        baselines=baseline_single)
            rescaledOcclusion[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_fit:
            attributions = torch.from_numpy(FIT.attribute(input, labels))
            rescaledFIT[idx:idx + batch_size, :, :] = Helper.givenAttGetRescaledSaliency(num_timesteps, num_features, attributions)

        if run_ifit:
            attributions = torch.from_numpy(inverse_fit_attribute(input, pretrained_model, ft_dim_last=True))
            rescaledIFIT[idx:idx + batch_size, :, :] = attributions

        if run_wfit:
            attributions = torch.from_numpy(wfit_attribute(input, pretrained_model, N=test_shape[1], ft_dim_last=True, single_label=True))
            rescaledWFIT[idx:idx + batch_size, :, :] = attributions

        if run_iwfit:
            attributions = torch.from_numpy(wfit_attribute(input, pretrained_model, N=test_shape[1], ft_dim_last=True, single_label=True, inverse=True))
            rescaledIWFIT[idx:idx + batch_size, :, :] = attributions

        idx += batch_size

    if run_grad:
        print("Saving Grad", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_Grad_rescaled", rescaledGrad)
    if run_grad_tsr:
        print("Saving Grad_TSR", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_Grad_TSR_rescaled", rescaledGrad_TSR)

    if run_ig:
        print("Saving IG", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_IG_rescaled", rescaledIG)
    if run_ig_tsr:
        print("Saving IG_TSR", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_IG_TSR_rescaled", rescaledIG_TSR)

    if run_dl:
        print("Saving DL", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_DL_rescaled", rescaledDL)

    if run_gs:
        print("Saving GS", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_GS_rescaled", rescaledGS)

    if run_dls:
        print("Saving DLS", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_DLS_rescaled", rescaledDLS)
    if run_dls_tsr:
        print("Saving DLS_TSR", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_DLS_TSR_rescaled", rescaledDLS_TSR)

    if run_sg:
        print("Saving SG", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_SG_rescaled", rescaledSG)

    if run_shapley_sampling:
        print("Saving ShapleySampling", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_ShapleySampling_rescaled",
                rescaledShapleySampling)

    if run_feature_permutation:
        print("Saving FeaturePermutation", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_FeaturePermutation_rescaled",
                rescaledFeaturePermutation)

    if run_feature_ablation:
        print("Saving FeatureAblation", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_FeatureAblation_rescaled",
                rescaledFeatureAblation)

    if run_occlusion:
        print("Saving Occlusion", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_Occlusion_rescaled", rescaledOcclusion)

    if run_fit:
        print("Saving FIT", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_FIT_rescaled", rescaledFIT)

    if run_ifit:
        print("Saving IFIT", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_IFIT_rescaled", rescaledIFIT)

    if run_wfit:
        print("Saving WFIT", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_WFIT_rescaled", rescaledWFIT)

    if run_iwfit:
        print("Saving IWFIT", model_name + "_" + model_type)
        np.save(saliency_dir + model_name + "_" + model_type + "_IWFIT_rescaled", rescaledIWFIT)



def main(args,DatasetsTypes,DataGenerationTypes,models,device):
    for m in range(len(models)):

        for x in range(len(DatasetsTypes)):
            for y in range(len(DataGenerationTypes)):

                if(DataGenerationTypes[y]==None):
                    args.DataName=DatasetsTypes[x]+"_Box"
                else:
                    args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]

                Training=np.load(args.data_dir+"SimulatedTrainingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TrainingMetaDataset=np.load(args.data_dir+"SimulatedTrainingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TrainingLabel=TrainingMetaDataset[:,0]

                Testing=np.load(args.data_dir+"SimulatedTestingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TestingDataset_MetaData=np.load(args.data_dir+"SimulatedTestingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TestingLabel=TestingDataset_MetaData[:,0]



                Training = Training.reshape(Training.shape[0],Training.shape[1]*Training.shape[2])
                Testing = Testing.reshape(Testing.shape[0],Testing.shape[1]*Testing.shape[2])


                scaler = MinMaxScaler()
                scaler.fit(Training)
                Training = scaler.transform(Training)
                Testing = scaler.transform(Testing)

                TrainingRNN = Training.reshape(Training.shape[0] , args.NumTimeSteps,args.NumFeatures)
                TestingRNN = Testing.reshape(Testing.shape[0] , args.NumTimeSteps,args.NumFeatures)



                train_dataRNN = data_utils.TensorDataset(torch.from_numpy(TrainingRNN), torch.from_numpy(TrainingLabel))
                train_loaderRNN = data_utils.DataLoader(train_dataRNN, batch_size=args.batch_size, shuffle=True)


                test_dataRNN = data_utils.TensorDataset(torch.from_numpy(TestingRNN),torch.from_numpy( TestingLabel))
                test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)
    


                modelName="Simulated"
                modelName+=args.DataName



                saveModelName="Models/"+models[m]+"/"+modelName
                saveModelBestName =saveModelName +"_BEST.pkl"



                pretrained_model = torch.load(saveModelBestName,map_location=device) 
                Test_Acc  =   checkAccuracy(test_loaderRNN , pretrained_model, args.NumTimeSteps, args.NumFeatures)
                print('{} {} model BestAcc {:.4f}'.format(args.DataName,models[m],Test_Acc))

                if Test_Acc >= 0:
                    run_saliency_methods(Helper.getSaliencyMethodsFromArgs(args), pretrained_model, TestingRNN.shape,
                                         train_loaderRNN, test_loaderRNN, device, models[m], modelName, args.Saliency_dir,
                                         args.Saliency_Maps_graphs_dir + '/TSR_attributions', [0, 10, 20, 30])

                else:
                    logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

                    logging.debug('{} {} model BestAcc {:.4f}'.format(args.DataName,models[m],Test_Acc))

                    if not os.path.exists(args.ignore_list):
                        with open(args.ignore_list, 'w') as fp: 
                            fp.write(args.DataName+'_'+models[m]+'\n')

                    else:
                        with open(args.ignore_list, "a") as fp:
                            fp.write(args.DataName+'_'+models[m]+'\n')

    if args.plot:
        getSaliencyMapMetadata(args.Saliency_dir, args.Saliency_Maps_graphs_dir, [0, 10, 20, 30])
