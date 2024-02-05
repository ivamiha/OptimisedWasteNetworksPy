import requests


def get_city_coords(city):
    """
    Obtain latitude and longitude coordinate pair for the provided city utilising the Nominatim geocoder.

    Parameters
    ----------
    city (str): name of the city of interest

    Returns
    -------
    output (list): list with coordinates in (lat, lng) format
    """

    # create url
    # url = "{0}{1}{2}".format(
    #    "http://nominatim.openstreetmap.org/search?city=",
    #    city,
    #    "&format=json&polygon=0",
    # )
    url = "{0}{1}{2}".format(
        "http://nominatim.openstreetmap.org/search?q=",
        city,
        "&format=json&polygon=0",
    )
    response = requests.get(url).json()[0]

    # parse response to list
    lst = [response.get(key) for key in ["lat", "lon"]]
    output = [float(i) for i in lst]
    return output


def get_country_coords(country, output_as="centre"):
    """
    Obtain latitude and longitude coordinate pairs for the provided country
    utilising the Nominatim geocoder.

    Parameters
    ----------
    country (str): name of the country of interest in English

    output_as (str): choose from 'centre' and 'bounding_box'
                     - 'centre' (default) for [lat_centre, lng_centre]
                     - 'bounding_box' for [lat_min, lat_max, lng_min, lng_max]

    Returns
    -------
    output (list): list with coordinates
    """

    # create url
    url = "{0}{1}{2}".format(
        "http://nominatim.openstreetmap.org/search?country=",
        country,
        "&format=json&polygon=0",
    )
    response = requests.get(url).json()[0]

    # parse response to list
    if output_as == "centre":
        lst = [response.get(key) for key in ["lat", "lon"]]
        output = [float(i) for i in lst]
    if output_as == "boundingbox":
        lst = response[output_as]
        output = [float(i) for i in lst]
    return output
