from geopy.distance import geodesic
import math


def coords_to_distance(coord, centre):
    """
    Calculates the straight-line Eucledian distance between the coordiantes ``coord`` and ``centre`` [km].

    Arguments
    ---------
    coord (list of floats): coordinate in the (lat, lng) format for which
    distance will be calculated

    centre (list of floats): coordinate in the (lat, lng) format from which
    distance will be calculated

    Returns
    -------
    distance (float): computed straight-line distance [km]
    """

    # compute straight-line distance using the geodesic formula
    distance = geodesic(coord, centre).km

    return distance


def coords_to_distances(coord, centre):
    """
    Calculates the x and y distances between the coordinates ``coord`` and
    ``centre`` [km].

    Arguments
    ---------
    coord (list of floats): coordinate in the (lat, lng) format for which
    distances will be calculated

    centre (list of floats): coordinate in the (lat, lng) format from which
    distances will be calculated

    Returns
    -------
    distances (list of floats): computed distances [km] in the (x_distance,
    y_distance) format
    """

    # compute straight-line distance using the geodesic formula
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
