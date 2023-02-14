import os, argparse, pickle
import torch

# Process:
#   1. Load from numpy
#   2. Convert to Torch
#   3. Break off validation split
#   4. Make times 
#   5. Make training dataset object
#   6. Package into dictionary
#   7. Save

class StateTrainDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

def print_tuple(t):
    print('X', t[0].shape)
    print('time', t[1].shape)
    print('y', t[2].shape)

def main(loc, split_no):
    # Load all data:
    for s in ['train', 'test']:
        # Importance:
        imp = pickle.load(open(os.path.join(loc, 'state_dataset_importance_{}.pkl'.format(s)), 'rb'))
        # imp already stored in matrix form, no need to convert
        log = pickle.load(open(os.path.join(loc, 'state_dataset_logits_{}.pkl'.format(s)), 'rb'))
        x = pickle.load(open(os.path.join(loc, 'state_dataset_x_{}.pkl'.format(s)), 'rb'))

        # Step 2: Convert to Torch
        xt = torch.from_numpy(x).permute(2, 0, 1)
        impt = torch.from_numpy(imp).permute(2, 0, 1)
        yt = torch.from_numpy((log > 0.5).astype(int)) # Convert logits to static
        yt = yt.sum(dim=-1)
    
        if s == 'train':
            Xtrain, ytrain = xt, yt

            # Step 3: break off validation set
            whole_inds = torch.randperm(Xtrain.shape[1])
            val_inds = whole_inds[:100]

            Xval = Xtrain[:,val_inds,:]
            yval = ytrain[val_inds]
            # Step 4: make times
            timeval = torch.arange(1,Xval.shape[0]+1).unsqueeze(-1).repeat(1,Xval.shape[1])

            Xtrain = Xtrain[:,whole_inds[100:],:]
            ytrain = ytrain[whole_inds[100:]]
            timetrain = torch.arange(1,Xtrain.shape[0]+1).unsqueeze(-1).repeat(1,Xtrain.shape[1])

        elif s == 'test':
            Xtest, ytest = xt, yt
            timetest = torch.arange(1,Xtest.shape[0]+1).unsqueeze(-1).repeat(1,Xtest.shape[1])
            gt_exps = impt # Only keep GT explanations for test split

    # Step 5: Make training dataset object
    train_dataset = StateTrainDataset(Xtrain, timetrain, ytrain)

    # Step 6: package into dictionary
    dataset = {
        'train_loader': train_dataset,
        'val': (Xval, timeval, yval),
        'test': (Xtest, timetest, ytest),
        'gt_exps': gt_exps,
    }

    print('\nTrain')
    print_tuple((Xtrain, timetrain, ytrain))

    print('\nVal')
    print_tuple((Xval, timeval, yval))

    print('\nTest')
    print_tuple((Xtest, timetest, ytest))


    # Step 7: save
    torch.save(dataset, '/home/owq978/TimeSeriesXAI/datasets/StateTrans/split={}.pt'.format(split_no))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type = int, required = True)
    args = parser.parse_args()

    main(loc = 'simulated_data', split_no = args.split)
