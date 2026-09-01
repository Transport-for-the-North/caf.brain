"""Generate geographic information that helps guide the image processing prior
to object detection training and inference."""

# Built-Ins
import logging
import os.path
from pathlib import Path

# Third Party
import geopandas as gpd
import pandas as pd

LOG = logging.getLogger(__name__)
OS_DATA_PATH = Path(os.getenv("OS_DATA_PATH", r"B:\TfN_object_detection\os_data.gpkg"))
TOWNS_AND_CITIES_PATH = Path(
    os.getenv("TOWNS_AND_CITIES_PATH", r"B:\TfN_object_detection\Towns_and_Cities")
)


def generate_user_location_shp_file(
    output_path: Path,
    user_prediction_location: Path | str | list[str],
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame that contains coordinates of junctions.

    Parameters
    ----------
    output_path:
        Path to output folder.
    user_prediction_location:
        This can be a path to a shapefile that contains a column called MOSA,
        LSOA OR LAD. This column would contain strings of the MSOAs, LSOAs or
        LADS you want to use. Alternatively, this can be a list of string(s) of
        city's you want to use. Examples:
            - A city name as a string (e.g., 'London')
            - A list of city names (e.g., ['London', 'Manchester'])
            - A path to a shapefile with LSOA, MSOA, or LAD columns
    Returns
    -------
    GeoDataFrame of points to process for prediction.
    """
    path_obj: Path | str | list[str] | None = None

    os_junc_path = output_path / "OS_junction_coordinates_final.shp"
    if os_junc_path.exists():
        LOG.info("Loading existing OS data from %s", os_junc_path)
        os_data = gpd.read_file(os_junc_path)
    else:
        os_path = OS_DATA_PATH
        if os.path.exists(os_path):
            LOG.info("Processing OS data from %s", os_path)
            os_data = _process_os(os_path=os_path, output_path=output_path)
        else:
            raise ValueError(f"Issue with {os_path}. It is not in the expected location")

    is_path = False
    if isinstance(user_prediction_location, str):
        candidate = Path(user_prediction_location)

        if candidate.exists() and candidate.suffix.lower() in {".shp", ".gpkg", ".geojson"}:
            is_path = True
            path_obj = candidate
        else:
            is_path = False
            path_obj = user_prediction_location

    elif isinstance(user_prediction_location, list):
        is_path = False
        path_obj = user_prediction_location

    else:
        raise ValueError(
            "user_prediction_location must be a string (city name), "
            "list of city names, or a valid shapefile path"
        )

    if isinstance(path_obj, list) or (isinstance(path_obj, str) and not is_path):
        LOG.info("Loading city boundaries for: %s", path_obj)
        cities_shp = TOWNS_AND_CITIES_PATH
        if os.path.exists(cities_shp):
            df = gpd.read_file(cities_shp)
            df = df[["TCITY15CD", "TCITY15NM", "geometry"]]
            df = df.rename(columns={"TCITY15NM": "CITY"})
        else:
            raise ValueError(f"Issue with {cities_shp}. It is not in the expected location")

        cities = [path_obj] if isinstance(path_obj, str) else path_obj
        df = df[df["CITY"].isin(cities)]

    elif is_path:
        LOG.info("Loading custom shapefile: %s", path_obj)
        df = gpd.read_file(path_obj)

        if "LSOA21CD" in df.columns:
            df = df[["LSOA21CD", "LSOA21NM", "geometry"]]
            df = df.rename(columns={"LSOA21CD": "LSOA"})
        elif "MSOA21CD" in df.columns:
            raise NotImplementedError("MSOA processing not yet implemented")
        elif "LAD21CD" in df.columns:
            df = df[["LAD21CD", "LAD21NM", "geometry"]]
            df = df.rename(columns={"LAD21CD": "LAD"})
        else:
            raise ValueError(
                "No valid location columns found. Expected one of: "
                "'LSOA21CD', 'LAD21CD', or 'MSOA21CD'."
            )

    else:
        raise ValueError(
            f"Invalid user_prediction_location type: {type(path_obj)}. "
            f"Must be a string (city name), list (city names), or valid shapefile path"
        )

    if df.empty:
        raise ValueError(
            f"No boundaries found for: {path_obj}. " f"Check your input is correct."
        )

    LOG.info("Loaded %d boundaries", len(df))

    LOG.info("Finding junctions within boundaries")
    joined = gpd.sjoin(os_data, df, how="inner", predicate="within")

    if joined.empty:
        raise ValueError(
            f"No junctions found within the specified boundaries.\n"
            f"OS data CRS: {os_data.crs}, Boundary CRS: {df.crs}"
        )

    LOG.info("Found %d junctions within boundaries", len(joined))

    out_juncs = output_path / "junctions_within_your_boundaries.shp"
    joined.to_file(out_juncs, index=False)
    LOG.info("Saved junction results to %s", out_juncs)

    return joined


def _process_os(os_path: Path, output_path: Path) -> gpd.GeoDataFrame:
    """
    Helper function for processing OS data.

    Parameters
    ----------
    os_path:
        Path to OS data.
    output_path:
        Path to output results.

    Returns
    -------
    Processed OS data as a GeoPandas DataFrame.
    """
    os0 = gpd.read_file(os_path, layer="road_node")
    os1 = gpd.read_file(os_path, layer="motorway_junction")
    os0 = os0[os0["form_of_road_node"] != "pseudo node"]
    os0 = os0[["id", "geometry"]]
    os1 = os1[["id", "geometry"]]
    LOG.info("Loaded %d road_node features (after removing pseudo nodes)", len(os0))
    LOG.info("Loaded %d motorway_junction features", len(os1))
    all_junctions = pd.concat([os0, os1], ignore_index=True)
    all_junctions = gpd.GeoDataFrame(all_junctions, geometry="geometry", crs=os0.crs)
    LOG.info("Total junction features combined: %d", len(all_junctions))
    duplicates = all_junctions.duplicated(subset=["id", "geometry"])
    LOG.info("Duplicate junction records detected: %d", duplicates.sum())

    out = output_path / "OS_junction_coordinates_final.shp"
    all_junctions.to_file(out)
    return all_junctions
