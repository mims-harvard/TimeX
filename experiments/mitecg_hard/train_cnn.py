import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.simple import CNN
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data import EpiDataset
from txai.utils.data.preprocess import process_MITECG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 2,
    epsilon = 1.0,
    weight = None,#torch.tensor([1.0, 3.0]),
    #weight =None,
)

for i in range(1, 6):
    torch.cuda.empty_cache()
    trainEpi, val, test, _ = process_MITECG(split_no = i, device = device, hard_split = True, normalize = False, 
        balance_classes = False, div_time = False, need_binarize = True, exclude_pac_pvc = True,
        base_path = 'datasets/drive/datasets_and_models/MITECG-Hard/')
    train_dataset = EpiDataset(trainEpi.X, trainEpi.time, trainEpi.y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)

    print(trainEpi.y)
    
    print('X shape')
    print(trainEpi.X.shape)
    print('y shape', trainEpi.y.shape)
    

    val = (val.X, val.time, val.y)
    test = (test.X, test.time, test.y)

    model = CNN(
        d_inp = val[0].shape[-1],
        n_classes = 2,
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    
    spath = 'models/MITECG-Hard_cnn_split={}.pt'.format(i)
    print('Saving at {}'.format(spath))

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 2,
        num_epochs = 200,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        validate_by_step = None,
        use_scheduler = False,
        print_freq = 1
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, spath)

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))