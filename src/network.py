import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as sp


def region_generator(length, n, customer_list):
    """
    Generates fictional region represented by a square whose sides measure ``length`` with ``n`` equidistant nodes placed along them. Eucledian distances utilised throughout.

    Parameters
    ----------
    length (float): length of a side of the square region [km]

    n (int): number of nodes along each side of the square region []

    customer_list (str): string consisting names of the customers in the region
    """

    # build x and y vectors for the region
    x = np.linspace(0, length, n)
    y = np.linspace(0, length, n)
    x_vector, y_vector = np.meshgrid(x, y)

    # convert vectors into coordinate points
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

    # create DataFrame containing domain and export it to a csv file
    domain = pd.DataFrame(points, columns=["xcord", "ycord"])
    domain.to_csv("domain_coordinates.csv", float_format="%.3f")

    # begin some plotting stuff... not entirely sure what is being done
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 18))
    domain.plot(ax=ax1, kind="scatter", x="xcord", y="ycord", color="k")
    ax1.set_xlabel("Horizontal distance [km]")
    ax1.set_ylabel("Vertical distance [km]")
    ax1.set_xlim(-10, length + 10)
    ax1.set_ylim(-10, length + 10)
    for k, v in domain.iterrows():
        ax1.annotate(k, v)

    # calculate distances
    D1 = pd.DataFrame(
        sp.distance_matrix(domain.values, domain.values), index=i_list, columns=j_list
    )
    D1.to_csv(r"D1.csv", float_format="%.3f")

    w, z = [], []
    for i in customer_list:
        w.append(domain.loc[i]["xcord"])
        z.append(domain.loc[i]["ycord"])

    w = np.array(w).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)

    customer = pd.DataFrame(
        np.concatenate((w, z), axis=1), columns=["xcord", "ycord"], index=customer_list
    )
    customer.to_csv("customer_coordinates.csv", float_format="%.3f")

    customer.plot(ax=ax2, kind="scatter", x="xcord", y="ycord", color="b")
    ax2.set_xlabel("Horizontal distance [km]")
    ax2.set_ylabel("Vertical distance [km]")
    ax2.set_xlim(-10, length + 10)
    ax2.set_ylim(-10, length + 10)
    for k, v in customer.iterrows():
        ax2.annotate(k, v)

    fig.savefig("source_sinks.pdf", dpi=1200)
    plt.show()

    n_list = []
    for num in customer_list:
        n_list.append("C_" + str(num))

    D2 = pd.DataFrame(
        sp.distance_matrix(domain.values, domain.values), index=j_list, columns=k_list
    )
    D2.to_csv("D2.csv", float_format="%.2f")
    D3 = pd.DataFrame(
        sp.distance_matrix(domain.values, domain.values), index=k_list, columns=l_list
    )
    D3.to_csv("D3.csv", float_format="%.2f")
    D4 = pd.DataFrame(
        sp.distance_matrix(domain.values, domain.values), index=l_list, columns=m_list
    )
    D4.to_csv("D4.csv", float_format="%.2f")
    D5 = pd.DataFrame(
        sp.distance_matrix(domain.values, customer.values), index=m_list, columns=n_list
    )
    D5.to_csv("D5.csv", float_format="%.2f")
