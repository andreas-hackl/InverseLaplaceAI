import torch
import invlap

def reconstruct_correlator(A, E, t_values):
    """
    Reconstructs the correlator from the amplitudes and energies.

    Parameters
    ----------
    A : torch.Tensor
        The amplitudes of the states.
    E : torch.Tensor
        The energies of the states.
    t_values : torch.Tensor
        The time values at which to evaluate the correlator.
    
    Returns
    -------
    torch.Tensor
        The reconstructed correlator.
    """
    
    t_grid = t_values.view(1, 1, -1)
    A_ = A.unsqueeze(2)
    E_ = E.unsqueeze(2)
    

    terms = A_ * torch.exp(-E_ * t_grid)
    return terms.sum(dim=1)




def weighted_mse_loss(C_pred, C_true, C_std):
    """
    Computes the weighted mean squared error loss between the predicted and true correlators.
    
    Parameters
    ----------
    C_pred : torch.Tensor
        The predicted correlator.
    C_true : torch.Tensor
        The true correlator.
    C_std : torch.Tensor
        The standard deviation of the true correlator.
    
    Returns
    -------
    torch.Tensor
        The computed loss.
    """

    loss = ((C_pred - C_true) / C_std)**2
    mask = torch.isfinite(loss)

    return loss[mask].mean()


def train(model, optimizer, data, Cstd, device, epochs=10):
    """
    Trains the model using the provided data and optimizer.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for training.
    data : np.ndarray
        The jackknife samples of the correlator.
    Cstd : np.ndarray
        The standard deviation of the correlator.
    device : torch.device
        The device to be used for training (CPU or GPU).
    batch_size : int
        The batch size for training.
    epochs : int, optional
        The number of epochs for training, by default 10.
    
    Returns
    -------
    tuple
        The trained model and the loss values.
    """

    X = torch.tensor(data, dtype=torch.float64).to(device)
    Nt = X.shape[1]
    t_values = torch.arange(Nt, dtype=torch.float64).to(device)

    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    C_std_t = torch.tensor(Cstd, dtype=torch.float64).to(device)

    loss_values = []

    for epoch in range(epochs):
        for i, (x,) in enumerate(dataloader):
            C_real = x.to(device)

            optimizer.zero_grad()

            A_pred, E_pred = model(x)
            C_pred = reconstruct_correlator(A_pred, E_pred, t_values)

            loss = weighted_mse_loss(C_pred, C_real, C_std_t)
            loss.backward()

            optimizer.step()
            loss_values.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        


    return model, loss_values






    