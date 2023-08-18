import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as sp


def region_generator(length, n, customers_list):
    """
    Generates fictional region represented by a square whose sides measure
    ``length`` with ``n`` equidistant nodes placed along them. Eucledian
    distances utilised throughout.

    Parameters
    ----------
    length (float): length of a side of the square region [km]

    n (int): number of nodes along each side of the square region []

    customers_list (arr): array with node number corresponding to customers in
    the region

    Returns
    -------
    D (arr): array containing DataFrames representing distances between
    facilities.
    """

    # build x and y vectors representing nodes in the region
    x = np.linspace(0, length, n)
    y = np.linspace(0, length, n)
    x_vector, y_vector = np.meshgrid(x, y)

    # convert vectors into coordinates and store in array ``points``
    points = []
    for i in range(n):
        for j in range(n):
            points.append([x_vector[i, j], y_vector[i, j]])

    # build list for each facility type containing indices of all nodes
    i_list, j_list, k_list, l_list, m_list = [], [], [], [], []
    for idx in range(0, n * n):
        i_list.append("S_" + str(idx))
        j_list.append("OCF_" + str(idx))
        k_list.append("MPF_" + str(idx))
        l_list.append("PF_" + str(idx))
        m_list.append("DPF_" + str(idx))
    # do the same for all nodes containing customers
    n_list = []
    for num in customers_list:
        n_list.append("C_" + str(num))

    # create DataFrame containing sources and export it to csv file
    sources = pd.DataFrame(points, columns=["xcord", "ycord"])
    sources.to_csv("coordinates_sources.csv", float_format="%.3f")

    # create DataFrame containing customers and export it to csv file
    w, z = [], []
    for i in customers_list:
        w.append(sources.loc[i]["xcord"])
        z.append(sources.loc[i]["ycord"])
    w = np.array(w).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    customers = pd.DataFrame(
        np.concatenate((w, z), axis=1), columns=["xcord", "ycord"], index=customers_list
    )
    customers.to_csv("coordinates_customers.csv", float_format="%.3f")

    # plot resulting fictional region
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    # plot sources in the fictional region
    sources.plot(ax=ax1, kind="scatter", x="xcord", y="ycord", color="k")
    ax1.set_xlabel("Horizontal distance [km]")
    ax1.set_ylabel("Vertical distance [km]")
    ax1.set_xlim(-length * 0.3, length * 1.3)
    ax1.set_ylim(-length * 0.3, length * 1.3)
    ax1.set_title("Source location")
    for k, v in sources.iterrows():
        ax1.annotate(k, v, textcoords="offset points", xytext=(0, 10), ha="center")
    # plot customers in fictional region
    customers.plot(ax=ax2, kind="scatter", x="xcord", y="ycord", color="b")
    ax2.set_xlabel("Horizontal distance [km]")
    ax2.set_ylabel("Vertical distance [km]")
    ax2.set_xlim(-length * 0.3, length * 1.3)
    ax2.set_ylim(-length * 0.3, length * 1.3)
    ax2.set_title("Customer location")
    for k, v in customers.iterrows():
        ax2.annotate(k, v, textcoords="offset points", xytext=(0, 10), ha="center")
    # save as vector graphics and show to user
    fig.savefig("sources_and_sinks.pdf", dpi=1200)
    plt.show()

    # calculate Eucledian distances between all point pairs & save to csv
    D1 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=i_list, columns=j_list
    )
    D1.to_csv(r"D1.csv", float_format="%.3f")
    D2 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=j_list, columns=k_list
    )
    D2.to_csv("D2.csv", float_format="%.2f")
    D3 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=k_list, columns=l_list
    )
    D3.to_csv("D3.csv", float_format="%.2f")
    D4 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=l_list, columns=m_list
    )
    D4.to_csv("D4.csv", float_format="%.2f")
    D5 = pd.DataFrame(
        sp.distance_matrix(sources.values, customers.values),
        index=m_list,
        columns=n_list,
    )
    D5.to_csv("D5.csv", float_format="%.2f")

    D = [D1, D2, D3, D4, D5]
    return D
