import numpy as np
import torch
import invlap
import qstatpy
import matplotlib.pyplot as plt

import invlap.qstatpy_extention

def main():
    # Example parameters
    T = np.arange(32)
    energies = np.array([0.6, 0.9, 1.2])
    overlaps = np.array([0.5, 0.3, 0.2])

    # Generate fake data
    Nc = 100
    data = invlap.fake_data.generate_fake_data(T, Nc, energies, overlaps, noise_factor=0.001)
    # Create a QStat database
    db = qstatpy.Database("example.json")
    
    # Add the generated data to the database
    for i in range(Nc):
        db.add_data(data[i,:], "Correlator", "RAW", f"c_{i}")

    db.jackknife(("Correlator", "RAW"), ("Correlator", "jk_mean"), lambda x: x)

    # Plot the Correlator

    fig, ax = plt.subplots()
    x, y, ys = db.curve("Correlator", "jk_mean")
    ax.errorbar(x, y, yerr=ys, marker='o', ls='', lw=0.8, fillstyle='none', capsize=5, markersize=5)
    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(r"$C(t)$")
    ax.set_yscale("log")

    for i, (E, A) in enumerate(zip(energies, overlaps)):
        ax.plot(T, A * np.exp(-E * T), lw=0.8, ls='--', color='black')

    ax.set_ylim(5e-9, 5)
    fig.savefig("correlator.png", dpi=300)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    jk_data = np.array([v for k, v in db.get_data("Correlator", "jk_mean").items()])

    model = invlap.model.MassNN(T.shape[0], len(energies)).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, loss_values = invlap.train.train(model, optimizer, jk_data, ys, device, epochs=100)

    fig, ax = plt.subplots()
    ax.plot(loss_values)
    ax.set_yscale("log")
    fig.savefig("loss.png", dpi=300)

    beta_expected = np.concatenate((overlaps, energies))
    beta_pred, beta_pred_std = invlap.qstatpy_extention.model_prediction(db, ("Correlator", "jk_mean"), ("Correlator", "beta"), model, device)

    for i in range(len(beta_expected)):
        print(f"Expected: {beta_expected[i]:.4f}, Predicted: {beta_pred[i]:.7f} ± {beta_pred_std[i]:.7f}")

    


    
if __name__ == "__main__":
    main()