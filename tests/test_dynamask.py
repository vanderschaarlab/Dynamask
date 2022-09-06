import torch
from dynamask.attribution.mask import Mask
from dynamask.attribution.perturbation import GaussianBlur
from dynamask.utils.losses import mse
from dynamask.attribution.mask_group import MaskGroup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

def test_sanity():
# Define a pseudo-black box:
    def black_box(input):
        output = input[-1, :]  # The black-box returns the features of the last time step
        return output
# Define a random input:
    X = torch.randn(10, 3).to(device) # The shape of the input has to be (T, N_features)

# Fit a mask to the input with a Gaussian Blur perturbation:
    pert = GaussianBlur(device)
    mask = Mask(pert, device)
    mask.fit(X, black_box, loss_function=mse, keep_ratio=0.1, size_reg_factor_init=0.01) # Select the 10% most important features
    assert mask.get_error() == 0

def test_groups():
# Define a pseudo-black box:
    def black_box(input):
        output = input[-1, :]  # The black-box returns the features of the last time step
        return output

# Define a random input:
    X = torch.randn(10, 3).to(device) # The shape of the input has to be (T, N_features)

# Fit a group of masks to the input with a Gaussian Blur perturbation:
    areas = [.1, .15, .2, .25] # These are the areas of the different masks
    pert = GaussianBlur(device)
    masks = MaskGroup(pert, device)
    masks.fit(X, black_box, loss_function=mse, area_list=areas, size_reg_factor_init=0.01)

# Extract the extremal mask:
    epsilon = 0.01
    mask = masks.get_extremal_mask(threshold=epsilon)

    assert mask.get_error() == 0
