import numpy as np
import torch

from FIT.TSX.generator import FeatureGenerator, train_feature_generator


def inverse_fit_attribute(x, model, activation=None, ft_dim_last=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    def model_predict(x):
        if ft_dim_last:
            x = x.permute(0, 2, 1)
        if activation is not None:
            return activation(model(x))
        else:
            return model(x)

    model.eval()

    if ft_dim_last:
        x = x.permute(0, 2, 1)

    batch_size, num_features, num_timesteps = x.shape
    score = torch.zeros(x.shape, device=device)

    for t in range(num_timesteps):
        p_y = model_predict(x[:, :, :t + 1])
        for f in range(num_features):
            x_hat = x[:, :, :t + 1].clone()
            x_hat[:, f, -1] = torch.mean(x)
            p_y_hat = model_predict(x_hat)
            div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
            score[:, f, t] = 2. / (1 + torch.exp(-5 * div)) - 1

    if ft_dim_last:
        score = score.permute(0, 2, 1)

    return score.detach().cpu().numpy()


def wfit_attribute(x, model, N, activation=None, ft_dim_last=False, single_label=False, collapse=True, inverse=False, generators=None, n_samples=10, cv=0):
    assert not single_label or not collapse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #if N == 1 and inverse:
    #    return inverse_fit_attribute(x, model, activation, ft_dim_last)

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    def model_predict(x):
        if ft_dim_last:
            x = x.permute(0, 2, 1)
        if activation is not None:
            return activation(model(x))
        else:
            return model(x)

    model.eval()

    if ft_dim_last:
        x = x.permute(0, 2, 1)

    batch_size, num_features, num_timesteps = x.shape
    scores = []

    start = num_timesteps - 1 if single_label else 0

    for t in range(start, num_timesteps):
        window_size = min(t, N)
        #score = torch.zeros(batch_size, num_features, window_size, device=device)
        score = np.zeros((batch_size, num_features, window_size))

        if t == 0:
            if ft_dim_last:
                score = score.permute(0, 2, 1)
            scores.append(score)
            continue

        p_y = model_predict(x[:, :, :t + 1])
        p_tm1 = model_predict(x[:, :, :t])

        for f in range(num_features):
            masked_f = [f] if inverse else list(set(range(num_features)) - {f})
            kl_div_expectations = []
            kl_div_temporal_expectations = []
            kl_div_unexplained_expectations = []

            for n in range(window_size):
                #print("Trace: ", t, f, n, window_size)
                x_hat = x[:, :, :t + 1].clone()
                if not inverse:
                    p_tmn = model_predict(x[:, :, :t-n])
                    temporal_difference = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_tmn), p_y), -1)
                    kl_div_temporal_expectations.append(temporal_difference.detach().cpu().numpy())
                
                div_all = []
                div_unexplained = []
                for _ in range(n_samples):
                    
                    if generators is None:
                        # Carry forward
                        x_hat[:, masked_f, t - n:t + 1] = x_hat[:, masked_f, t - n - 1, None]
                    else:
                        for mask_f in masked_f:
                            x_hat[:, mask_f, t - n:t + 1] = generators[mask_f](x_hat[:, :, t - n:t + 1], x_hat[:, :, :t - n])[0][:, :n + 1]

                    p_y_hat = model_predict(x_hat)

                    if inverse:
                        div = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
                    else:
                        unexplained_difference = torch.sum(torch.nn.KLDivLoss(reduction='none')(torch.log(p_y_hat), p_y), -1)
                        div = temporal_difference - unexplained_difference
                        div_unexplained.append(unexplained_difference.detach().cpu().numpy())

                    div_all.append(div.detach().cpu().numpy())          
 
                E_div = np.mean(np.array(div_all), axis=0)
                kl_div_expectations.append(E_div)
                kl_div_unexplained_expectations.append(np.mean(np.array(div_unexplained), axis=0))
                
                acc_score = 2. / (1 + np.exp(-5 * E_div)) - 1
                if n > 0:
                    if inverse:
                        kl = 2. / (1 + np.exp(-5 * kl_div_expectations[n-1])) - 1
                        score[:, f, window_size - n - 1] = acc_score - kl # - kl_div_expectations[n-1]
                    else:
                        prev_score = score[:, f, window_size - n]
                        #prev_score = score[:, f, window_size - n:].sum(axis=-1)
                        E_div = (kl_div_temporal_expectations[n] - kl_div_temporal_expectations[n-1]) - (kl_div_unexplained_expectations[n] - kl_div_unexplained_expectations[n-1])
                        #E_div = (kl_div_temporal_expectations[n] - kl_div_temporal_expectations[n-1]) - kl_div_unexplained_expectations[n]
                        #E_div = (kl_div_temporal_expectations[n]) - (kl_div_unexplained_expectations[n] - kl_div_unexplained_expectations[n-1])
                        acc_score =  2. / (1 + np.exp(-5 * E_div)) - 1
                        #unexplained = 2. / (1 + np.exp(-5 * )) - 1
                        score[:, f, window_size - n - 1] = acc_score
                        #score[:, f, window_size - n - 1] = acc_score - prev_score

                else:
                    score[:, f, window_size - n - 1] = acc_score
                #score[:, f, window_size - n - 1] = acc_score - score[:, f, window_size - n] if n > 0 else acc_score


        if ft_dim_last:
            #score = score.permute(0, 2, 1)
            score = np.transpose(score, (0, 2, 1))

        scores.append(score)


    if single_label:
        scores = scores[0]
    elif collapse:
        #scores = max_collapse(scores)
        scores = absmax_collapse(scores)
        #scores = mean_collapse(scores)

    return scores


def absmax_collapse(attributions):
    combined_attrs = np.zeros((attributions[0].shape[0], attributions[0].shape[1], len(attributions)))
    for pred in range(len(attributions)):
        attributions[pred] = np.nan_to_num(attributions[pred])
        start = pred - attributions[pred].shape[-1] + 1
        end = pred + 1
        combined_attrs[:, :, start:end] = np.where(np.abs(combined_attrs[:, :, start:end]) > np.abs(attributions[pred]),
                                                   combined_attrs[:, :, start:end], attributions[pred])
    return combined_attrs


def max_collapse(attributions):
    combined_attrs = np.zeros((attributions[0].shape[0], attributions[0].shape[1], len(attributions)))
    for pred in range(len(attributions)):
        attributions[pred] = np.nan_to_num(attributions[pred])
        start = pred - attributions[pred].shape[-1] + 1
        end = pred + 1
        combined_attrs[:, :, start:end] = np.where(combined_attrs[:, :, start:end] > attributions[pred],
                                                   combined_attrs[:, :, start:end], attributions[pred])
    return combined_attrs

def mean_collapse(attributions):
    combined_attrs = np.zeros((attributions[0].shape[0], attributions[0].shape[1], len(attributions)))
    for pred in range(len(attributions)):
        attributions[pred] = np.nan_to_num(attributions[pred])
        start = pred - attributions[pred].shape[-1] + 1
        end = pred + 1
        combined_attrs[:, :, start:end] = np.where(np.abs(combined_attrs[:, :, start:end]) > np.abs(attributions[pred]),
                                                   combined_attrs[:, :, start:end], attributions[pred])
    return combined_attrs

def get_wfit_generators(train_loader, test_loader, N, name, train, cv=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_features = next(iter(train_loader))[0].shape[1]
    generators = []
    for f in range(num_features):
        generator = FeatureGenerator(num_features, hist=True, hidden_size=50, prediction_size=N, data=name, conditional=False)
        if train:
            train_feature_generator(generator, train_loader, test_loader, 'feature_generator', n_epochs=300,  feature_to_predict=f, cv=cv)
        generator.load_state_dict(torch.load(f'ckpt/{name}/{f}_feature_generator_{cv}.pt'))
        generator.to(device)
        generators.append(generator)
    
    return generators

        
        
        
