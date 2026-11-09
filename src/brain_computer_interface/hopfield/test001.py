import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np


    return (np,)


@app.cell
def _(np):
    from hopfieldkit import HopfieldNetwork

    societe_apaisee = np.array([1, 1, 1, 1])
    fracture_nordsud = np.array([1, 1, -1, -1])

    net = HopfieldNetwork(n_units=4).fit([societe_apaisee, fracture_nordsud])

    corrupted = np.array([-1, -1, 1, -1])
    recovered, energies = net.recall(corrupted, order="sequential", record_energy=True)
    recovered, energies
    return


@app.cell
def _(np):
    from brain_computer_interface.hopfield.hopfield import HopfieldNetwork
    import matplotlib.pyplot as plt

    societe_apaisee = np.array([1, 1, 1, 1])
    fracture_nordsud = np.array([1, 1, -1, -1])

    net = HopfieldNetwork(number_of_neuron=4)
    net.fit([societe_apaisee, fracture_nordsud])

    # corrupted = np.array([-1, 1, -1, -1])
    corrupted = np.array([-1, -1, 1, -1])
    recovered = net.recall(corrupted)
    print(recovered)  # -> [1, 1, -1, -1], fracture_nordsud recovered
    print(net.energies)
    # plt.plot(net.energies, net.energies)
    return


if __name__ == "__main__":
    app.run()
