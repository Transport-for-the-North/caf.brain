from pathlib import Path
import pandas as pd
import geopandas as gpd
import logging
LOG = logging.getLogger(__name__)


# def _get_general_area_junc_coords(output_path: Path,
#                                   os_path: Path | None = None,
#                                   user_prediction_location: Path | str | list[str] | None = None,
#                                   locations: str | None = None) -> gpd.GeoDataFrame:
#
#     if user_prediction_location is None:
#         raise ValueError("user_prediction_location cannot be None. Please provide \n"
#                          "either a string to a town / city location or a csv path. \n"
#                          "If providing a csv path, ensure a column called MOSA, LSOA or LAD is present \n"
#                          "and contains strings that are either MOSAs, LSOAs or LADs \n"
#                          "(LAD, MOSA, LSOA).")
#
#     os_junc_path = output_path / "OS_junction_coordinates_final.shp"
#
#     if os_junc_path.exists():
#         LOG.info("Loading existing OS data from %s", os_junc_path)
#         os_data = gpd.read_file(os_junc_path)
#     else:
#         if os_path is None:
#             raise FileNotFoundError(
#                 f"OS junction data not found at {os_junc_path} and no os_path provided to generate it.\n"
#                 f"Please provide os_path to process the data."
#             )
#         LOG.info("Processing OS data from %s", os_path)
#         os_data = _process_os(os_path=os_path, output_path=output_path)
#
#     is_path = False
#     if isinstance(user_prediction_location, str):
#         try:
#             path_obj = Path(user_prediction_location)
#             is_path = path_obj.exists() and path_obj.suffix in ['.shp', '.gpkg', '.geojson']
#         except (ValueError, OSError):
#             is_path = False
#
#     if isinstance(user_prediction_location, list) or (isinstance(user_prediction_location, str) and not is_path):
#         df = gpd.read_file(r"T:\Adil Zaheer\Object Detection - extra data for prediction\cities\Major_Towns_and_Cities_(December_2015)_Boundaries_V2.shp")
#         df = df[['TCITY15CD', 'TCITY15NM', 'geometry']]
#         df = df.rename(columns={'TCITY15NM': 'CITY'})
#         cities = [user_prediction_location] if isinstance(user_prediction_location, str) else user_prediction_location
#         df = df[df["CITY"].isin(cities)]
#
#     elif is_path:
#         if locations is None:
#             raise ValueError("If providing a path for user_prediction_location, \n"
#                              "locations cannot be None. Make sure this column is in \n"
#                              "your data. It must contain strings of either LSOAs, MSOAs or LADs. \n"
#                              "Based on what is present, that should be the name of the column."
#                              )
#         df = gpd.read_file(user_prediction_location)
#
#         if 'LSOA' in df.columns:
#             df = df[['LSOA21CD', 'LSOA21NM', 'geometry']]
#             df = df.rename(columns={'LSOA21CD': 'LSOA'})
#
#         elif 'MSOA' in df.columns:
#             pass  # TODO
#
#         elif 'LAD' in df.columns:
#             df = df[['LAD21CD', 'LAD21NM', 'geometry']]
#             df = df.rename(columns={'LAD21CD': 'LAD'})
#
#         else:
#             raise ValueError("Please ensure your data has a column called LSOA or LAD.\n"
#                              "You need to assign this to the locations argument")
#
#     else:
#         raise ValueError(
#             f"Invalid user_prediction_location type: {type(user_prediction_location)}\n"
#             f"Must be a string (city name), list (city names), or valid shapefile path"
#         )
#
#     if df.empty:
#         raise ValueError(
#             f"No boundaries found for: {user_prediction_location}\n"
#             f"Check your input is correct"
#         )
#
#     joined = gpd.sjoin(os_data, df, how="inner", predicate="within")
#     LOG.info("Found %d junctions within boundaries", len(joined))
#
#     out_juncs = output_path / "junctions_within_your_boundaries.shp"
#     joined.to_file(out_juncs, index=False)
#
#     return joined


def _get_general_area_junc_coords(
        output_path: Path,
        os_path: Path | None = None,
        user_prediction_location: Path | str | list[str] | None = None,
        locations: str | None = None
) -> gpd.GeoDataFrame:
    """
    Generating a GeoDataFrame that contains coordinates of
    junctions to predict.

    Parameters
    ----------
    output_path: Path to output folder.
    os_path:
    user_prediction_location: This can be a path to a shapefile that
                              contains a column called MOSA, LSOA OR LAD.
                              This column would contain strings of the
                              MSOAs, LSOAs or LADS you want to use. Alternatively,
                              this can be a list of string(s) of
                              city's you want to use.
    locations: This corresponds to user_prediction_location if a path is
               provided. This will be a column title in the shapefile
               called either LSOA, MSOA or LAD. This must contain the
               zones you want to predict.

    Returns
    -------
    GeoDataFrame of points to process for prediction.
    """
    if user_prediction_location is None:
        raise ValueError(
            "user_prediction_location cannot be None. Please provide either:\n"
            "  - A city name as a string (e.g., 'London')\n"
            "  - A list of city names (e.g., ['London', 'Manchester'])\n"
            "  - A path to a shapefile with LSOA, MSOA, or LAD columns"
        )

    # Load or process OS junction data
    os_junc_path = output_path / "OS_junction_coordinates_final.shp"
    if os_junc_path.exists():
        LOG.info("Loading existing OS data from %s", os_junc_path)
        os_data = gpd.read_file(os_junc_path)
    else:
        if os_path is None:
            raise FileNotFoundError(
                f"OS junction data not found at {os_junc_path} and no os_path provided.\n"
                f"Please provide os_path to generate it."
            )
        LOG.info("Processing OS data from %s", os_path)
        os_data = _process_os(os_path=os_path, output_path=output_path)

    is_path = False
    if isinstance(user_prediction_location, str):
        try:
            path_obj = Path(user_prediction_location)
            is_path = path_obj.exists() and path_obj.suffix in ['.shp', '.gpkg', '.geojson']
        except (ValueError, OSError):
            is_path = False

    if isinstance(user_prediction_location, list) or (isinstance(user_prediction_location, str) and not is_path):
        LOG.info("Loading city boundaries for: %s", user_prediction_location)
        cities_shp = r"T:\Adil Zaheer\Object Detection - extra data for prediction\cities\Major_Towns_and_Cities_(December_2015)_Boundaries_V2.shp"
        df = gpd.read_file(cities_shp)
        df = df[['TCITY15CD', 'TCITY15NM', 'geometry']]
        df = df.rename(columns={'TCITY15NM': 'CITY'})

        cities = [user_prediction_location] if isinstance(user_prediction_location, str) else user_prediction_location
        df = df[df["CITY"].isin(cities)]

    elif is_path:
        LOG.info("Loading custom shapefile: %s", user_prediction_location)
        if locations is None:
            raise ValueError(
                "When providing a shapefile path, 'locations' parameter cannot be None.\n"
                "Please specify the column name: 'LSOA', 'MSOA', or 'LAD'"
            )

        df = gpd.read_file(user_prediction_location)

        if locations == 'LSOA':
            df = df[['LSOA21CD', 'LSOA21NM', 'geometry']]
            df = df.rename(columns={'LSOA21CD': 'LSOA'})
        elif locations == 'MSOA':
            raise NotImplementedError("MSOA processing not yet implemented")
        elif locations == 'LAD':
            df = df[['LAD21CD', 'LAD21NM', 'geometry']]
            df = df.rename(columns={'LAD21CD': 'LAD'})
        else:
            raise ValueError(
                f"Invalid locations value: '{locations}'. Must be 'LSOA', 'MSOA', or 'LAD'"
            )

    else:
        raise ValueError(
            f"Invalid user_prediction_location type: {type(user_prediction_location)}. "
            f"Must be a string (city name), list (city names), or valid shapefile path"
        )

    if df.empty:
        raise ValueError(
            f"No boundaries found for: {user_prediction_location}. "
            f"Check your input is correct."
        )

    LOG.info("Loaded %d boundaries", len(df))

    # Spatial join to find junctions within boundaries
    LOG.info("Finding junctions within boundaries...")
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


def _process_os(os_path: Path,
                output_path: Path) -> gpd.GeoDataFrame:
    """
    Helper function for processing OS data.

    Parameters
    ----------
    os_path: Path to OS data.
    output_path: Path to output results.

    Returns
    -------
    Processed OS data as a GeoPandas DataFrame.
    """
    os0 = gpd.read_file(os_path, layer='road_node')
    os1 = gpd.read_file(os_path, layer='motorway_junction')
    os0 = os0[os0["form_of_road_node"] != "pseudo node"]
    os0 = os0[['id', 'geometry']]
    os1 = os1[['id', 'geometry']]
    print(len(os0))
    print(len(os1))
    all_junctions = pd.concat([os0, os1], ignore_index=True)
    all_junctions = gpd.GeoDataFrame(all_junctions, geometry="geometry", crs=os0.crs)
    print(len(all_junctions))
    duplicates = all_junctions.duplicated(subset=["id", "geometry"])
    print(duplicates.sum())

    out = output_path / "OS_junction_coordinates_final.shp"
    all_junctions.to_file(out)
    return all_junctions
