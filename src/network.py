import numpy as np
import pandas as pd
import scipy.spatial as sp
import matplotlib.pyplot as plt
import utils.get_coords as gc
import utils.convert_coords as cc


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
    facilities
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
        l_list.append("CPF_" + str(idx))
        m_list.append("DPF_" + str(idx))
    # do the same for all nodes containing customers
    n_list = []
    for num in customers_list:
        n_list.append("C_" + str(num))

    # create DataFrame containing sources and export it to csv file
    sources = pd.DataFrame(points, columns=["xcord", "ycord"])
    sources.to_csv("results/coordinates_sources.csv", float_format="%.3f")

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
    customers.to_csv("results/coordinates_customers.csv", float_format="%.3f")

    # plot resulting fictional region
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    # plot sources in the fictional region
    sources.plot(ax=ax1, kind="scatter", x="xcord", y="ycord", color="k")
    ax1.set_xlabel("Horizontal distance [km]")
    ax1.set_ylabel("Vertical distance [km]")
    ax1.set_xlim(-length * 0.3, length * 1.3)
    ax1.set_ylim(-length * 0.3, length * 1.3)
    ax1.set_title("Source locations")
    for k, v in sources.iterrows():
        ax1.annotate(k, v, textcoords="offset points", xytext=(0, 10), ha="center")
    # plot customers in fictional region
    customers.plot(ax=ax2, kind="scatter", x="xcord", y="ycord", color="b")
    ax2.set_xlabel("Horizontal distance [km]")
    ax2.set_ylabel("Vertical distance [km]")
    ax2.set_xlim(-length * 0.3, length * 1.3)
    ax2.set_ylim(-length * 0.3, length * 1.3)
    ax2.set_title("Customer locations")
    for k, v in customers.iterrows():
        ax2.annotate(k, v, textcoords="offset points", xytext=(0, 10), ha="center")
    # save as vector graphics and show to user
    fig.savefig("results/sources_and_sinks.pdf", dpi=1200)
    plt.show()

    # calculate Eucledian distances between all point pairs & save to csv
    D1 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=i_list, columns=j_list
    )
    D1.to_csv(r"results/D1.csv", float_format="%.3f")
    D2 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=j_list, columns=k_list
    )
    D2.to_csv("results/D2.csv", float_format="%.2f")
    D3 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=k_list, columns=l_list
    )
    D3.to_csv("results/D3.csv", float_format="%.2f")
    D4 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=l_list, columns=m_list
    )
    D4.to_csv("results/D4.csv", float_format="%.2f")
    D5 = pd.DataFrame(
        sp.distance_matrix(sources.values, customers.values),
        index=m_list,
        columns=n_list,
    )
    D5.to_csv("results/D5.csv", float_format="%.2f")

    # return distance matrices in array format
    D = [D1, D2, D3, D4, D5]
    return D


def region_builder(sources_coords, customers_list, country=None, img_path=None):
    """
    Build the region that is defined by the coordinates in ``sources_coords``
    and ``customer_coords``. Eucledian distances are measured using kilometeres
    as units.

    Parameters
    ----------
    sources_coords (arr): array consisting of [x_coord, y_coord] entries for the
    sources within the region

    customers_coords (arr): array consisting of [x_coord, y_coord] entries for
    the customers within the region

    country (str): optional string containing name of the considered country in
    english, used to obtain the country's centre and extremes (northmost,
    southmost, eastmost, westmost) for setting plot limits

    img_path (str): optional string containing the location of the image file
    which (if specified) will be used as a background for the generated plot, REQUIRES ``country`` to be specified as well

    Returns
    -------
    D (arr): array containing DataFrames representing distances between
    facilities
    """

    # build list for each facility type containing indices of all nodes
    i_list, j_list, k_list, l_list, m_list = [], [], [], [], []
    for idx in range(0, len(sources_coords)):
        i_list.append("S_" + str(idx))
        j_list.append("OCF_" + str(idx))
        k_list.append("MPF_" + str(idx))
        l_list.append("CPF_" + str(idx))
        m_list.append("DPF_" + str(idx))
    # do the same for all nodes containing customers
    n_list = []
    for num in customers_list:
        n_list.append("C_" + str(num))

    # create DataFrame containing sources and export it to csv file
    sources = pd.DataFrame(sources_coords, columns=["xcord", "ycord"])
    sources.to_csv("results/coordinates_sources.csv", float_format="%.3f")

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
    customers.to_csv("results/coordinates_customers.csv", float_format="%.3f")

    # obtain plot limits using geocoder if ``country`` is specified
    if country != None:
        # obtain corresponding lats and lngs using Nominatim geocoder
        centre = gc.get_country_coords(country)
        bounding_box = gc.get_country_coords(country, output_as="boundingbox")
        # calculate x and y distances to the bottom left and top right corners
        bottom_left = cc.coords_to_distances(
            (bounding_box[0], bounding_box[2]), (centre[0], centre[1])
        )
        top_right = cc.coords_to_distances(
            (bounding_box[1], bounding_box[3]), (centre[0], centre[1])
        )
        # extract the required extents for the plot
        x_min = bottom_left[0]
        x_max = top_right[0]
        y_min = bottom_left[1]
        y_max = top_right[1]
    # otherwise, obtain plot limits based on plotted data
    else:
        x_vals = [sub_array[0] for sub_array in sources_coords]
        y_vals = [sub_array[1] for sub_array in sources_coords]
        length_x = max(x_vals) - min(x_vals)
        length_y = max(y_vals) - min(y_vals)

    # plot resulting region
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    # plot sources in the region
    sources.plot(ax=ax1, kind="scatter", x="xcord", y="ycord", color="k")
    ax1.set_xlabel("Horizontal distance [km]")
    ax1.set_ylabel("Vertical distance [km]")
    ax1.set_title("Source locations")
    for k, v in sources.iterrows():
        ax1.annotate(
            k, v, textcoords="offset points", xytext=(0, 10), ha="center", zorder=1
        )
    # plot customers in region
    customers.plot(ax=ax2, kind="scatter", x="xcord", y="ycord", color="b")
    ax2.set_xlabel("Horizontal distance [km]")
    ax2.set_ylabel("Vertical distance [km]")
    ax2.set_title("Customer locations")
    for k, v in customers.iterrows():
        ax2.annotate(
            k, v, textcoords="offset points", xytext=(0, 10), ha="center", zorder=1
        )
    # set corresponding limits to both plots
    if country != None:
        ax1.set_xlim(x_min * 1.2, x_max * 1.2)
        ax1.set_ylim(y_min * 1.2, y_max * 1.2)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
    else:
        ax1.set_xlim(min(x_vals) - length_x * 0.2, max(x_vals) + length_x * 0.2)
        ax1.set_ylim(min(y_vals) - length_y * 0.2, max(y_vals) + length_y * 0.2)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
    # set background image if ``img_path`` has been provided
    if img_path != None and country != None:
        background_img = plt.imread(img_path)
        ax1.imshow(background_img, zorder=0, extent=[x_min, x_max, y_min, y_max])
        ax2.imshow(background_img, zorder=0, extent=[x_min, x_max, y_min, y_max])
    # save as vector graphics and show to user
    fig.savefig("results/sources_and_sinks.pdf", dpi=1200)
    plt.show()

    # calculate Eucledian distances between all point pairs & save to csv
    D1 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=i_list, columns=j_list
    )
    D1.to_csv(r"results/D1.csv", float_format="%.3f")
    D2 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=j_list, columns=k_list
    )
    D2.to_csv("results/D2.csv", float_format="%.2f")
    D3 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=k_list, columns=l_list
    )
    D3.to_csv("results/D3.csv", float_format="%.2f")
    D4 = pd.DataFrame(
        sp.distance_matrix(sources.values, sources.values), index=l_list, columns=m_list
    )
    D4.to_csv("results/D4.csv", float_format="%.2f")
    D5 = pd.DataFrame(
        sp.distance_matrix(sources.values, customers.values),
        index=m_list,
        columns=n_list,
    )
    D5.to_csv("results/D5.csv", float_format="%.2f")

    # return distance matrices in array format
    D = [D1, D2, D3, D4, D5]
    return D
