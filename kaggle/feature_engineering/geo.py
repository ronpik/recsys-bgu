from typing import Sequence, Any, Tuple

import pandas as pd
import geopy

RELEVANT_COUNTRIES = {"US", "CA"}

def get_region_coordinate(country: str, region_str) -> Tuple[float, float]:
    api = geopy.Nominatim()



def get_country_region_value(country: str, region_str) -> str:
    if country not in RELEVANT_COUNTRIES:
        return country

    return country


def get_geo_features(sample: pd.Series) -> Tuple:
    country = sample.country_code
    region = sample.region

    country_region_str = get_country_region_value(country, region)
    coordinates = get_region_coordinate(country, region)


