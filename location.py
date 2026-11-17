import urllib.parse


def get_location(location):

    # Check location availability
    if not isinstance(location, dict):
        return "Location not provided"

    latitude = location.get("latitude")
    longitude = location.get("longitude")

    # Validate coordinates
    if latitude is None or longitude is None:
        return "Location not provided"

    search_query = "nearby pharmacies"

    encoded_query = urllib.parse.quote(search_query)

    map_link = (
        f"https://www.google.com/maps/search/"
        f"{encoded_query}/@{latitude},{longitude},15z"
    )

    return map_link