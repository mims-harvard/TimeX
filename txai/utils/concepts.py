import torch
from dataclasses import dataclass, field
from txai.utils.functional import transform_to_attn_mask

class PreChosenConceptList:
    '''
    Class to hold the sample and mask for a pre-chosen prototypical concept

    sample: (T, Nc, d) sample
    mask: (T, Nc, d) binary mask (float)
    '''
    def __init__(self, X, times, masks):
        #assert X.shape == masks.shape, 'Sample and mask shape must match'
        self.X = X
        self.times = times
        self.masks = masks
        self.attn_masks = transform_to_attn_mask(self.masks) # Output is (B, T, T)
        self.Nc = masks.shape[1] # Batch size
    
    def __getitem__(self, ind):
        return self.X[:,ind,:], self.times[:,ind], self.attn_masks[ind,:,:]

    def get_all(self, attn_mask = True):
        if attn_mask:
            return self.X, self.times, self.attn_masks
        else:
            return self.X, self.times, self.masks

    def to(self, device):
        self.X = self.X.to(device)
        self.times = self.times.to(device)
        self.attn_masks = self.attn_masks.to(device)

        self.X.requires_grad = False
        self.times.requires_grad = False
        self.attn_masks.requires_grad = False

def sample_shift_matrices(X, expected_shift):
    pass

def batch_shift_X_tensor(X, s):
    """
    From ChatGPT ------ works out of the box!

    Shift a PyTorch tensor along dimension 0 for each batch by a different amount.
    Positive values of s shift right, negative values shift left.
    
    Args:
    - X: input tensor of size (T, B, d)
    - s: tensor of integers of length B, indicating the amount to shift each batch along dimension 0
    
    Returns:
    - shifted tensor of size (T, B, d)
    """
    T, B, d = X.size()
    s = s.unsqueeze(0).unsqueeze(-1) # shape: (1, B, 1)
    shifts = (torch.arange(T).reshape(T, 1, 1) - s) % T # shape: (T, B, 1)
    return torch.gather(X, 0, shifts.expand(T, B, d).long().to(X.device))

def batch_shift_time_tensor(times, s):
    """
    From ChatGPT ------ works out of the box!

    Shift a PyTorch tensor along dimension 0 for each batch by a different amount.
    Positive values of s shift right, negative values shift left.
    
    Args:
    - X: input tensor of size (T, B, d)
    - s: tensor of integers of length B, indicating the amount to shift each batch along dimension 0
    
    Returns:
    - shifted tensor of size (T, B, d)
    """
    T, B = times.size()
    s = s.unsqueeze(0) # shape: (1, B, 1)
    shifts = (torch.arange(T).reshape(T, 1) - s) % T # shape: (T, B, 1)
    return torch.gather(times, 0, shifts.expand(T, B).long().to(times.device))

def sample_shift_indices(p, N, max_abs_shift = None):
    """
    From ChatGPT --- 
    Sample from a two-sided geometric distribution with parameter p.
    Returns a tensor of N random integers from the set {..., -2, -1, 0, 1, 2, ...} with probability mass function
    P(X = k) = (1-p)^(|k|-1) * p, where |k| is the absolute value of k.
    """
    k = torch.empty(N, dtype=torch.long)
    is_left = torch.bernoulli(torch.ones(N, dtype=torch.float32)*0.5).bool()
    k[is_left] = -torch.distributions.Geometric(p).sample((is_left.sum(),)).to(torch.long)
    k[~is_left] = torch.distributions.Geometric(p).sample(((~is_left).sum(),)).to(torch.long)

    if max_abs_shift is not None:
        k = torch.where(k > max_abs_shift, max_abs_shift, k)
        k = torch.where(k < (-1.0 * max_abs_shift), (-1.0 * max_abs_shift), k)

    return k

default_augmentation_selections = {
    'gaussian': True,
    'time_shift': True,
    'mask_shift': True,
    'random_masking': True,
}

@dataclass
class GaussianBlurParams:
    std: float = field(default = 1.0)

@dataclass
class TimeShiftParams:
    p: float = field(default = 0.5)
    max_abs_shift: int = field(default = 20)

@dataclass
class MaskShiftParams:
    p: float = field(default = 0.5)
    max_abs_shift: int = field(default = 20)

@dataclass
class RandomMaskParams:
    p: float = field(default = 0.9) # Larger values = less random masking out

class ConceptsWithAugmentations:

    def __init__(self, 
            X, 
            times, 
            masks, 
            concept_ref = None, 
            gaussian_blur_params: GaussianBlurParams = None,
            time_shift_params: TimeShiftParams = None,
            mask_shift_params: MaskShiftParams = None,
            random_mask_params: RandomMaskParams = None,
        ):

        self.X = X
        self.times = times
        self.masks = masks
        self.attn_masks = transform_to_attn_mask(self.masks) # Output is (B, T, T)
        self.Nc = masks.shape[1] # Batch size

        # IMPLEMENT LATER
        self.concept_ref = concept_ref
        
        self.gaussian_blur_params = gaussian_blur_params
        self.time_shift_params = time_shift_params
        self.mask_shift_params = mask_shift_params
        self.random_mask_params = random_mask_params

    def to(self, device):
        self.X = self.X.to(device)
        self.times = self.times.to(device)
        self.attn_masks = self.attn_masks.to(device)
        self.masks = self.masks.to(device)

        # self.X.requires_grad = False
        # self.times.requires_grad = False
        # self.attn_masks.requires_grad = False
        # self.masks.requires_grad = False
        
    def __getitem__(self, ind):
        return self.X[:,ind,:], self.times[:,ind], self.attn_masks[ind,:,:]

    def get_all_original(self, attn_mask = True):
        if attn_mask:
            return self.X, self.times, self.attn_masks
        else:
            return self.X, self.times, self.masks

    def get_all_augmentations(self, n_aug_per_concept, attn_mask = True, reg_mask = False):
        # Generate augmentations on-the-fly

        X_all, times_all, masks_all = self.X.clone(), self.times.clone(), self.masks.clone() # Clone each tensor to avoid overwriting

        X_aug_list = []
        times_aug_list = []
        attn_masks_aug_list = []
        masks_aug_list = []

        for i in range(self.Nc):

            X, times, masks = X_all[:,i,:].unsqueeze(1), times_all[:,i].unsqueeze(1), masks_all[:,i,:].unsqueeze(1)

            # Repeat:
            X = X.repeat(1, n_aug_per_concept, 1)
            times = times.repeat(1, n_aug_per_concept)
            masks = masks.repeat(1, n_aug_per_concept, 1)

            if self.gaussian_blur_params is not None:
                G = torch.randn_like(X).to(X.device) * self.gaussian_blur_params.std # Just multiply, a la reparameterization trick
                X = X + G
            else:
                G = torch.zeros_like(X).to(X.device)
            
            if self.time_shift_params is not None:
                shift_inds = sample_shift_indices(
                    p = self.time_shift_params.p, 
                    N = X.shape[1],
                    max_abs_shift = self.time_shift_params.max_abs_shift)

                times = batch_shift_time_tensor(times, shift_inds) # Just shift times - equivalent to shifting X and mask
            
            if self.mask_shift_params is not None:
                shift_inds = sample_shift_indices(
                    p = self.mask_shift_params.p, 
                    N = X.shape[1],
                    max_abs_shift = self.mask_shift_params.max_abs_shift)
                # Also shift mask
                masks = batch_shift_X_tensor(masks, shift_inds)
            
            if self.random_mask_params is not None:
                p_tensor = torch.full_like(masks, self.random_mask_params.p)
                random_mask = torch.bernoulli(p_tensor).to(masks.device)

                # Essentially an AND operation
                masks = masks * random_mask

            attn_masks = transform_to_attn_mask(masks)

            X_aug_list.append(X)
            times_aug_list.append(times)
            attn_masks_aug_list.append(attn_masks)
            masks_aug_list.append(masks)

        X_aug = torch.stack(X_aug_list, dim = 0) # Shape (Nc, T, n_aug_per_concept, d)
        times_aug = torch.stack(times_aug_list, dim = 0) # Shape: (Nc, T, n_aug_per_concept)
        attn_masks_aug = torch.stack(attn_masks_aug_list) # Shape (Nc, n_aug_per_concept, T, T)
        masks_aug = torch.stack(masks_aug_list) # Stacks along dim = 0 -> (Nc, T, n_aug_per_concept, d_z)

        if attn_mask and reg_mask:
            return X_aug, times_aug, attn_masks_aug, masks_aug
        elif attn_mask:
            return X_aug, times_aug, attn_masks_aug
        else:
            return X_aug, times_aug, masks_aug # IF both false, returns only regular mask

if __name__ == '__main__':
    # Test the shifting operator:

    print(sample_shift_indices(p = 0.2, N = 5, max_abs_shift = 5))

    # a = torch.randn(5, 2, 1)
    # print('a', a)
    # st = batch_shift_tensor(a, s = torch.tensor([1, -2]))
    
    # print('Index 0')
    # print(a[:,0,0])
    # print(st[:,0,0])

    # print('Index 1')
    # print(a[:,1,0])
    # print(st[:,1,0])
