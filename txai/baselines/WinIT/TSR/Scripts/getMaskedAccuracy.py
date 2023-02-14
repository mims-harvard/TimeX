import torch
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
import os
import time

from .Plotting.plot import *
from . import Helper
from .Helper import checkAccuracy


def get_masked_accuracy(saliency_methods, masked_acc_dir, data_name, model_type, model_name, device, test_loader,
                        num_timesteps, num_features, mask_dir, test_data, test_label, scaler, graph_dir, batch_size,
                        data_generation_process, sampler="regular", frequency=2.0,
                        kernel="Matern", ar_param=0.9, order=10, has_noise=False, plot=False):
    maskedPercentages = [i for i in range(0, 101, 10)]

    start = time.time()
    resultFileName = masked_acc_dir + data_name + "_" + model_type

    Y_DimOfGrid = len(maskedPercentages) + 1
    X_DimOfGrid = len(saliency_methods)

    Grid = np.zeros((X_DimOfGrid, Y_DimOfGrid), dtype='object')

    Grid[:, 0] = saliency_methods
    columns = ["saliency method"]
    for mask in maskedPercentages:
        columns.append(str(mask))

    savemodel_name = "Models/" + model_type + "/" + model_name
    saveModelBestName = savemodel_name + "_BEST.pkl"

    pretrained_model = torch.load(saveModelBestName, map_location=device)
    Test_Unmasked_Acc = checkAccuracy(test_loader, pretrained_model, num_timesteps, num_features)

    for s, saliency in enumerate(saliency_methods):
        Test_Masked_Acc = Test_Unmasked_Acc
        for i, maskedPercentage in enumerate(maskedPercentages):

            start_percentage = time.time()
            if (maskedPercentage == 0):
                Grid[s][i + 1] = Test_Unmasked_Acc
            elif (Test_Masked_Acc == 0):
                Grid[s][i + 1] = Test_Masked_Acc
            else:
                if (maskedPercentage != 100):
                    mask = np.load(mask_dir + model_name + "_" + model_type + "_" + saliency + "_" + str(
                        maskedPercentage) + "_percentSal_rescaled.npy")

                    toMask = np.copy(test_data)
                    MaskedTesting = Helper.maskData(data_generation_process, num_timesteps, num_features, sampler,
                                                      frequency, kernel, ar_param, order, has_noise, toMask, mask, True)
                    MaskedTesting = scaler.transform(MaskedTesting)
                    MaskedTesting = MaskedTesting.reshape(-1, num_timesteps, num_features)

                else:

                    MaskedTesting = np.zeros((test_data.shape[0], num_timesteps * num_features))
                    sample = Helper.generateNewSample(data_generation_process, num_timesteps, num_features, sampler,
                                                      frequency, kernel, ar_param, order, has_noise).reshape(num_timesteps * num_features)
                    MaskedTesting[:, :] = sample

                    MaskedTesting = scaler.transform(MaskedTesting)
                    MaskedTesting = MaskedTesting.reshape(-1, num_timesteps, num_features)

                if plot:
                    randomIndex = 10
                    plotExampleBox(MaskedTesting[randomIndex], graph_dir + data_name + "_" + model_type
                                   + "_" + saliency + "_percentMasked" + str(maskedPercentage), flip=True)

                Maskedtest_dataRNN = data_utils.TensorDataset(torch.from_numpy(MaskedTesting),
                                                              torch.from_numpy(test_label))
                Maskedtest_loader = data_utils.DataLoader(Maskedtest_dataRNN, batch_size=batch_size,
                                                             shuffle=False)

                Test_Masked_Acc = checkAccuracy(Maskedtest_loader, pretrained_model, num_timesteps,
                                                num_features)
                print('{} {} {} Acc {:.2f} Masked Acc {:.2f} Highest Value mask {}'.format(data_name, model_type,
                                                                                           saliency,
                                                                                           Test_Unmasked_Acc,
                                                                                           Test_Masked_Acc,
                                                                                           maskedPercentage))
                Grid[s][i + 1] = Test_Masked_Acc
            end_percentage = time.time()
    end = time.time()
    print('{} {} time: {}'.format(data_name, model_type, end - start))
    print()

    for percent in maskedPercentages:
        resultFileName = resultFileName + "_" + str(percent)
    resultFileName = resultFileName + "_percentSal_rescaled.csv"
    Helper.save_intoCSV(Grid, resultFileName, col=columns)


def main(args,DatasetsTypes,DataGenerationTypes,models,device):
    if  os.path.exists(args.ignore_list):
        f = open(args.ignore_list, 'r+')
        ignore_list = [line for line in f.readlines()]
        f.close()
        for i in range(len(ignore_list)):
            if('\n' in ignore_list[i]):
                ignore_list[i]=ignore_list[i][:-1]
    else:
        ignore_list=[]

    Saliency_Methods = Helper.getSaliencyMethodsFromArgs(args)
    Saliency_Methods.append("Random")

    for x in range(len(DatasetsTypes)):
        for y in range(len(DataGenerationTypes)):
            args.DataGenerationProcess=DataGenerationTypes[y]
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
            raw_Testing=np.copy(Testing)

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

            # save np.load
            np_load_old = np.load

            # modify the default parameters of np.load
            np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


            modelName="Simulated"
            modelName+=args.DataName

            for m in range(len(models)):
                if (args.DataName + "_" + models[m] in ignore_list):
                    print("ignoring", args.DataName + "_" + models[m])
                    continue
                
                get_masked_accuracy(Saliency_Methods, args.Masked_Acc_dir, args.DataName, models[m], modelName, device,
                                    test_loaderRNN, args.NumTimeSteps, args.NumFeatures, args.Mask_dir, raw_Testing,
                                    TestingLabel, scaler, args.Graph_dir, args.batch_size, args.DataGenerationProcess,
                                    args.Sampler, args.Frequency, args.Kernal, args.ar_param, args.Order, args.hasNoise,
                                    args.plot)

            np.load = np_load_old
