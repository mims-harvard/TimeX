import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

from .Helper import checkAccuracy
from .Models.LSTM import LSTM
from .Models.Transformer import Transformer
from .Models.LSTMWithInputCellAttention import LSTMWithInputCellAttention
from .Models.TCN import TCN


def train_model(model, model_type, model_name, criterion, train_loader, test_loader, device, num_timesteps,
				num_features, num_epochs, data_name, learning_rate):
	model.double()
	optimizerTimeAtten = torch.optim.Adam(model.parameters(), lr=learning_rate)

	saveModelName = "Models/" + model_type + "/" + model_name
	saveModelBestName = saveModelName + "_BEST.pkl"
	saveModelLastName = saveModelName + "_LAST.pkl"

	total_step = len(train_loader)
	Train_acc_flag = False
	Train_Acc = 0
	Test_Acc = 0
	BestAcc = 0
	BestEpochs = 0
	patience = 200

	for epoch in range(num_epochs):
		noImprovementflag = True
		for i, (samples, labels) in enumerate(train_loader):

			model.train()
			samples = Variable(samples)
			labels = labels.to(device)
			labels = Variable(labels).long()

			outputs = model(samples)
			loss = criterion(outputs, labels)

			optimizerTimeAtten.zero_grad()
			loss.backward()
			optimizerTimeAtten.step()

			if (i + 1) % 3 == 0:
				Test_Acc = checkAccuracy(test_loader, model, num_timesteps, num_features)
				Train_Acc = checkAccuracy(train_loader, model, num_timesteps, num_features)
				if (Test_Acc > BestAcc):
					BestAcc = Test_Acc
					BestEpochs = epoch + 1
					torch.save(model, saveModelBestName)
					noImprovementflag = False

				print(
					'{} {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.2f}, Test Accuracy {:.2f},BestEpochs {},BestAcc {:.2f} patience {}'
					.format(data_name, model_type, epoch + 1, num_epochs, i + 1, total_step, loss.item(),
							Train_Acc, Test_Acc, BestEpochs, BestAcc, patience))
			if (Train_Acc >= 99 or BestAcc >= 99):
				torch.save(model, saveModelLastName)
				Train_acc_flag = True
				break

		if (noImprovementflag):
			patience -= 1
		else:
			patience = 200

		if (epoch + 1) % 10 == 0:
			torch.save(model, saveModelLastName)

		if Train_acc_flag or patience == 0:
			break

		Train_Acc = checkAccuracy(train_loader, model, num_timesteps, num_features)
		print('{} {} BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(data_name, model_type, BestEpochs, BestAcc, Train_Acc))
	return saveModelBestName


def main(args,DatasetsTypes,DataGenerationTypes,models,device):
	criterion = nn.CrossEntropyLoss()
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


				if(models[m]=="LSTM"):
					net=LSTM(args.NumFeatures, args.hidden_size,args.num_classes,args.rnndropout).to(device)
				elif(models[m]=="LSTMWithInputCellAttention"):
					net=LSTMWithInputCellAttention(args.NumFeatures, args.hidden_size,args.num_classes,args.rnndropout,args.attention_hops,args.d_a).to(device)
				elif(models[m]=="Transformer"):
					net=Transformer(args.NumFeatures, args.NumTimeSteps, args.n_layers, args.heads, args.rnndropout,args.num_classes,time=args.NumTimeSteps).to(device)
				elif(models[m]=="TCN"):
					num_chans = [args.hidden_size] * (args.levels - 1) + [args.NumTimeSteps]
					net=TCN(args.NumFeatures,args.num_classes,num_chans,args.kernel_size,args.rnndropout).to(device)
				else:
					raise Exception(f'Model type {models[m]} not recognized')

				train_model(net, models[m], modelName, criterion, train_loaderRNN, test_loaderRNN, device,
							args.NumTimeSteps, args.NumFeatures, args.num_epochs, args.DataName, args.learning_rate)
