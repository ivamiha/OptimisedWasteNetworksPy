import numpy as np
import pandas as pd
import scipy.spatial as sp
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import math


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


def region_setup(sources_list, customers_list, img_path, region):
    """
    Sets up region which is specified via the provided arguments. TODO:
    Currently utilises Eucledian distances, upgrade to driving distances.

    Parameters
    ----------
    sources (list): list containing source coordiantes following the (lat, long)
    format

    customers (list): list containing customer coordinates following the (lat,
    long) format

    img_path (string): path to image containing map which is to be set as a
    background for the plot of the region (e.g. "images/map.png")

    region (string): name of the region which will be used by geocoder to
    determine center and extremes of the region for the proper alignment of the
    background image (e.g. "Germany")

    Returns
    -------
    D (arr): array containing DataFrames representing distances between
    facilities.
    """

    # build list for each facility type containing indices of all nodes
    i_list, j_list, k_list, l_list, m_list = [], [], [], [], []
    for idx in range(0, len(sources_list)):
        i_list.append("S_" + str(idx))
        j_list.append("OCF_" + str(idx))
        k_list.append("MPF_" + str(idx))
        l_list.append("CPF_" + str(idx))
        m_list.append("DPF_" + str(idx))
    # do the same for all nodes containing customers
    n_list = []
    for idx in range(0, len(customers_list)):
        n_list.append("C_" + str(idx))

    # obtain the centres and edges of the provided region using Nominatim
    geolocator = Nominatim(user_agent="192.168.1.117")
    country = geolocator.geocode(region)
    c_lat = country.latitude
    c_lng = country.longitude
    country = geolocator.geocode(region, exactly_one=False, limit=1)
    bounding_box = country[0].raw["boundingbox"]
    southmost = float(bounding_box[0])
    northmost = float(bounding_box[1])
    westmost = float(bounding_box[2])
    eastmost = float(bounding_box[3])
    # convert edge coordinates into distances from centre_coords [km]
    bottom_left = coord_to_dist((southmost, westmost), (c_lat, c_lng))
    top_right = coord_to_dist((northmost, eastmost), (c_lat, c_lng))
    southmost = bottom_left[1]
    northmost = top_right[1]
    westmost = bottom_left[0]
    eastmost = top_right[0]

    # convert source coordinates into distances from region center, save in
    # ``sources_plotting`` DF, NOTE: use ONLY for plotting, NOT computing
    names = ["xcord", "ycord"]
    sources_plotting = pd.DataFrame(sources_list, columns=names, dtype="float64")
    distances = [
        coord_to_dist((lat, lng), (c_lat, c_lng))
        for lat, lng in zip(sources_plotting["xcord"], sources_plotting["ycord"])
    ]
    sources_plotting = pd.DataFrame(distances, columns=names, dtype="float64")
    sources_plotting.to_csv("coordinates_sources.csv", float_format="%.3f")
    # do the same for the customer coordiantes with ``customers_plotting`` DF
    customers_plotting = pd.DataFrame(customers_list, columns=names, dtype="float64")
    distances = [
        coord_to_dist((lat, lng), (c_lat, c_lng))
        for lat, lng in zip(customers_plotting["xcord"], customers_plotting["ycord"])
    ]
    customers_plotting = pd.DataFrame(distances, columns=names, dtype="float64")
    customers_plotting.to_csv("coordinates_customers.csv", float_format="%.3f")

    # read background image (if user passed its directory)
    background_image = plt.imread(img_path)
    # plot provided region and display it to the user
    fig, ax = plt.subplots(figsize=(8, 8))
    # plot sources and customers in the provided region
    ax.scatter(
        sources_plotting["xcord"],
        sources_plotting["ycord"],
        c="black",
        marker="o",
        label="sources",
        zorder=1,
    )
    ax.scatter(
        customers_plotting["xcord"],
        customers_plotting["ycord"],
        c="blue",
        marker="o",
        label="customers",
        zorder=1,
    )
    # place the background image
    ax.imshow(
        background_image,
        zorder=0,
        extent=[
            westmost,
            eastmost,
            southmost,
            northmost,
        ],
    )
    # plot the customers in the provided region
    ax.set_xlabel("Horizontal distance [km]")
    ax.set_ylabel("Vertical distance [km]")
    ax.set_xlim(westmost - 25, eastmost + 25)
    ax.set_ylim(southmost - 25, northmost + 25)
    ax.legend(loc="upper right")
    # save as vector graphics and show to user
    fig.savefig("sources_and_sinks.pdf", dpi=1200)
    plt.show()

    # convert source coordinates into distances from first node, save in
    # ``sources_computing`` DF, NOTE: use ONLY for computing, NOT plotting
    sources_computing = pd.DataFrame(sources_list, columns=names, dtype="float64")
    first_node_coords = sources_list[0]
    fn_lat = first_node_coords[0]
    fn_lng = first_node_coords[1]
    distances = [
        coord_to_dist((lat, lng), (fn_lat, fn_lng))
        for lat, lng in zip(sources_computing["xcord"], sources_computing["ycord"])
    ]
    sources_computing = pd.DataFrame(distances, columns=names, dtype="float64")
    sources_computing.to_csv("distances_sources.csv", float_format="%.3f")
    # do the same for the customer coordiantes with ``customers_computing`` DF
    customers_computing = pd.DataFrame(customers_list, columns=names, dtype="float64")
    distances = [
        coord_to_dist((lat, lng), (fn_lat, fn_lng))
        for lat, lng in zip(customers_computing["xcord"], customers_computing["ycord"])
    ]
    customers_computing = pd.DataFrame(distances, columns=names, dtype="float64")
    customers_computing.to_csv("distances_customers.csv", float_format="%.3f")

    # calculate Eucledian distances between all point pairs in supply chain
    D1 = pd.DataFrame(
        sp.distance_matrix(sources_computing.values, sources_computing.values),
        index=i_list,
        columns=j_list,
    )
    D1.to_csv(r"D1.csv", float_format="%.3f")
    D2 = pd.DataFrame(
        sp.distance_matrix(sources_computing.values, sources_computing.values),
        index=j_list,
        columns=k_list,
    )
    D2.to_csv("D2.csv", float_format="%.2f")
    D3 = pd.DataFrame(
        sp.distance_matrix(sources_computing.values, sources_computing.values),
        index=k_list,
        columns=l_list,
    )
    D3.to_csv("D3.csv", float_format="%.2f")
    D4 = pd.DataFrame(
        sp.distance_matrix(sources_computing.values, sources_computing.values),
        index=l_list,
        columns=m_list,
    )
    D4.to_csv("D4.csv", float_format="%.2f")
    D5 = pd.DataFrame(
        sp.distance_matrix(sources_computing.values, customers_computing.values),
        index=m_list,
        columns=n_list,
    )
    D5.to_csv("D5.csv", float_format="%.2f")

    D = [D1, D2, D3, D4, D5]
    return D


@staticmethod
def coord_to_dist(coord, centre):
    """
    Calculates longitudinal and latitudinal distances between the coordiante
    ``cord`` and the ``centre`` of the provided region [km].

    Arguments
    ---------
    coord (list of floats): coordiante in the (latitude, longitude) format for
    which distances will be calculated

    centre (list of floats): coordinate in the (latitude, longitude) format from
    which distances will be calculated

    Returns
    -------
    distances (list of floats): computed distances [km] in the
    (longitudinal_distance, latitudinal distance), that is(x_distance,
    y_distance), format
    """
    # compute straight-line distance using geodesic formula
    distance = geodesic(coord, centre).km

    # unpack coordinates and convert them form degrees into radians
    lat1 = math.radians(centre[0])
    lat2 = math.radians(coord[0])
    lng1 = math.radians(centre[1])
    lng2 = math.radians(coord[1])

    # calculate the difference in longitude
    dlng = lng2 - lng1

    # calculate the bearing
    y = math.sin(dlng) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlng
    )
    bearing = math.atan2(y, x)
    # convert the bearing from radians to degrees
    bearing = math.degrees(bearing)

    # compute distance components and pack them into distances list
    x_distance = distance * math.sin(math.radians(bearing))
    y_distance = distance * math.cos(math.radians(bearing))
    distances = (x_distance, y_distance)

    return distances
