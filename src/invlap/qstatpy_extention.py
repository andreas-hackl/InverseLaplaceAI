import qstatpy
import numpy as np
import torch

def numpy_torch_wrapper(x, model, device):
    xt = torch.tensor(x).view(1, -1).to(device)
    A, E = model(xt)
    A_ = A.cpu().detach().numpy()[0]
    E_ = E.cpu().detach().numpy()[0]
    beta = np.concatenate((A_, E_))
    return beta

def model_prediction(db, in_tag, out_tag, model, device):
    db.estimate(in_tag, out_tag, lambda x: numpy_torch_wrapper(x, model, device))

    _, y, ys = db.curve(*out_tag)
    return y, ys


