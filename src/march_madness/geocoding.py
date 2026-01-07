"""Functions for looking up latitude, longitude, and altitude of cities using the Google Geocoding API."""

import os
import time
from typing import Literal

import polars as pl
import requests
from pydantic import BaseModel

from march_madness.loader import DataConfig, DataLoader
from march_madness.settings import DATA_DIR

STATUS = Literal[
    "OK",
    "ZERO_RESULTS",
    "OVER_DAILY_LIMIT",
    "OVER_QUERY_LIMIT",
    "REQUEST_DENIED",
    "INVALID_REQUEST",
    "UNKNOWN_ERROR",
]
LOCATION_TYPE = Literal[
    "ROOFTOP",
    "RANGE_INTERPOLATED",
    "GEOMETRIC_CENTER",
    "APPROXIMATE",
]


class Location(BaseModel):
    lat: float
    lng: float


class Bounds(BaseModel):
    northeast: Location
    southwest: Location


class AddressComponent(BaseModel):
    long_name: str
    short_name: str
    types: list[str]


class Geometry(BaseModel):
    bounds: Bounds | None = None
    location: Location
    location_type: LOCATION_TYPE
    viewport: Bounds


class GeocodeResult(BaseModel):
    address_components: list[AddressComponent]
    formatted_address: str
    geometry: Geometry
    place_id: str
    types: list[str]


class ElevationResult(BaseModel):
    elevation: float
    location: Location
    resolution: float


class GeocodeResponse(BaseModel):
    results: list[GeocodeResult]
    status: STATUS


class ElevationResponse(BaseModel):
    results: list[ElevationResult]
    status: STATUS


def fetch_geocode(city: str, state: str, api_key: str | None = None) -> GeocodeResponse:
    """Fetch geocode information for a given city and state."""
    params = {
        "key": api_key or os.environ.get("GOOGLE_GEOCODING_API_KEY"),
        "address": f"{city}, {state}",
    }
    response = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params=params,
    )
    return GeocodeResponse.model_validate(response.json())


def fetch_elevation(lat: float, lng: float, api_key: str | None = None) -> ElevationResponse:
    """Fetch elevation information for a given latitude and longitude."""
    params = {
        "key": api_key or os.environ.get("GOOGLE_GEOCODING_API_KEY"),
        "locations": f"{lat},{lng}",
    }
    response = requests.get(
        "https://maps.googleapis.com/maps/api/elevation/json",
        params=params,
    )
    return ElevationResponse.model_validate(response.json())


def geocode_cities() -> None:
    """Main function to geocode cities and save results to CSV."""
    data_loader = DataLoader(league="M")
    data_config = DataConfig()
    cities = data_loader.load_data(data_config.cities)

    cities_geo_list: list[dict] = []

    for row in cities.iter_rows(named=True):
        # First, hit the geocode API to get latitude and longitude
        city = row["City"]
        state = row["State"]
        geocode_model = fetch_geocode(
            city=city,
            state=state,
        )
        time.sleep(0.1)

        # Second, hit the elevation API to get the altitude
        lat = geocode_model.results[0].geometry.location.lat
        lng = geocode_model.results[0].geometry.location.lng
        elevation_model = fetch_elevation(
            lat=lat,
            lng=lng,
        )
        time.sleep(0.1)

        cities_geo_list.append(
            {
                "city_id": row["CityID"],
                "city": city,
                "state": state,
                "lat": lat,
                "lng": lng,
                "elevation": elevation_model.results[0].elevation,
            }
        )

    pl.DataFrame(cities_geo_list).write_csv(DATA_DIR / "CitiesGeocoded.csv")
