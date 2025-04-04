# Parameters for the IMT MSS-DC topology.
from dataclasses import dataclass, field
import numpy as np
import typing
from pathlib import Path
import geopandas as gpd

from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_orbit import ParametersOrbit

SHARC_ROOT_DIR = (Path(__file__) / ".." / ".." / ".." / "..").resolve()

@dataclass
class ParametersSelectActiveSatellite(ParametersBase):
    @dataclass
    class ParametersLatLongInsideCountry(ParametersBase):
        country_shapes_filename: Path = SHARC_ROOT_DIR / "sharc"/"topology"/"countries"/"ne_110m_admin_0_countries.shp"

        # may load automatically for different shapefiles
        __ALLOWED_COUNTRY_NAMES = [
            'Fiji', 'Tanzania', 'Western Sahara', 'Canada', 'United States of America', 'Kazakhstan',
            'Indonesia', 'Argentina', 'Chile', 'Democratic Republic of the Congo', 'Somalia', 'Kenya',
            'Haiti', 'Dominican Republic', 'Russia', 'The Bahamas', 'Falkland Islands', 'Norway', 'Greenland',
            'East Timor', 'South Africa', 'Lesotho', 'Mexico', 'Uruguay', 'Brazil', 'Bolivia', 'Peru',
            'Colombia', 'Panama', 'Costa Rica', 'Nicaragua', 'Honduras', 'El Salvador', 'Guatemala',
            'Venezuela', 'Guyana', 'Suriname', 'France', 'Ecuador', 'Puerto Rico', 'Jamaica', 'Cuba',
            'Zimbabwe', 'Botswana', 'Namibia', 'Senegal', 'Mali', 'Mauritania', 'Benin', 'Niger', 'Sudan',
            'Nigeria', 'Cameroon', 'Togo', 'Ghana', 'Ivory Coast', 'Guinea', 'Guinea-Bissau', 'Liberia',
            'Sierra Leone', 'Burkina Faso', 'Central African Republic', 'Republic of the Congo', 'Gabon',
            'Mozambique', 'Eswatini', 'Angola', 'Burundi', 'Israel', 'Lebanon', 'Madagascar', 'Palestine',
            'The Gambia', 'Tunisia', 'Algeria', 'Jordan', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Iraq',
            'Oman', 'Vanuatu', 'Cambodia', 'Thailand', 'Laos', 'Myanmar', 'Vietnam', 'North Korea', 'Zambia',
            'South Korea', 'Mongolia', 'India', 'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan', 'Afghanistan',
            'Tajikistan', 'Kyrgyzstan', 'Turkmenistan', 'Iran', 'Syria', 'Armenia', 'Sweden', 'Belarus',
            'Ukraine', 'Poland', 'Austria', 'Hungary', 'Moldova', 'Romania', 'Lithuania', 'Latvia', 'Belize',
            'Estonia', 'Germany', 'Bulgaria', 'Greece', 'Turkey', 'Albania', 'Croatia', 'Switzerland', 'Chad',
            'Luxembourg', 'Belgium', 'Netherlands', 'Portugal', 'Spain', 'Ireland', 'New Caledonia',
            'New Zealand', 'Australia', 'Sri Lanka', "People's Republic of China", 'Taiwan', 'Italy', 'Denmark',
            'Iceland', 'Azerbaijan', 'Georgia', 'Philippines', 'Malaysia', 'Brunei', 'Slovenia', 'Finland',
            'Slovakia', 'Czech Republic', 'Eritrea', 'Japan', 'Paraguay', 'Yemen', 'Saudi Arabia', 'Antarctica',
            'Turkish Republic of Northern Cyprus', 'Cyprus', 'Morocco', 'Egypt', 'Libya', 'Ethiopia', 'Djibouti',
            'Uganda', 'Rwanda', 'Bosnia and Herzegovina', 'North Macedonia', 'Serbia', 'Montenegro', 'Kosovo',
            'French Southern and Antarctic Lands', 'South Sudan', 'Trinidad and Tobago', 'Equatorial Guinea',
            'Malawi', 'United Kingdom', 'Somaliland', 'Solomon Islands', 'Papua New Guinea', 'Uzbekistan',
        ]

        __ALLOWED_COORDINATE_REFERENCES = [
            # it is a WGS84 based coordinate system that only contains (lat, long) information
            # which should be more than enough to describe any country borders or desired shape
            "EPSG:4326"
        ]

        country_name: str = None
        # margin from inside of border [km]
        # if positive, makes border smaller by x km
        # if negative, makes border bigger by x km
        margin_from_border: float = 0.0

        def validate(self, ctx):
            if self.country_name is None:
                raise ValueError(
                    f"{ctx}.country_name was not set, but is needed!"
                )

            if self.country_shapes_filename != ParametersLatLongInsideCountry.country_shapes_filename:
                f = gpd.read_file(self.country_shapes_filename, columns=["NAME"])
                if f.geometry.crs not in self.__ALLOWED_COORDINATE_REFERENCES:
                    raise ValueError(
                        f"Shapefile at {ctx}.country_shapes_filename = {self.country_shapes_filename}\n"
                        f"does not use one of the allowed formats {self.__ALLOWED_COORDINATE_REFERENCES},"
                        "with points as (lat, long).\n"
                        "If for some reason you really want to use another projection for this parameter\n"
                        "Add the projection so that this error isn't triggered"
                    )
                if "NAME" not in f:
                    raise ValueError(
                        f"Shapefile at {ctx}.country_shapes_filename = {self.country_shapes_filename}\n"
                        "does not contains a 'NAME' column, so it cannot be read"
                    )
                self.__ALLOWED_COUNTRY_NAMES = list(f["NAME"])

            if self.country_name not in self.__ALLOWED_COUNTRY_NAMES:
                raise ValueError(
                    f"{ctx}.country_name == {self.country_name} but shapefile only contains data on\n"
                    f"{self.__ALLOWED_COUNTRY_NAMES}"
                )

    __ALLOWED_CONDITIONS = [
        "LAT_LONG_INSIDE_COUNTRY",
        "MINIMUM_ELEVATION_FROM_ES",
    ]

    conditions: list[typing.Literal[
        "LAT_LONG_INSIDE_COUNTRY",
        "MINIMUM_ELEVATION_FROM_ES",
    ]] = field(default_factory=lambda: list([""]))

    minimum_elevation_from_es: float = None

    lat_long_inside_country: ParametersLatLongInsideCountry = field(default_factory=ParametersLatLongInsideCountry)

    def validate(self, ctx):
        if "LAT_LONG_INSIDE_COUNTRY" in self.conditions:
            self.lat_long_inside_country.validate(f"{ctx}.lat_long_inside_country")

        if "MINIMUM_ELEVATION_FROM_ES" in self.conditions:
            if not isinstance(self.minimum_elevation_from_es, float) and not isinstance(self.minimum_elevation_from_es, int):
                raise ValueError(
                    f"{ctx}.minimum_elevation_from_es is not a number!"
                )
            if self.minimum_elevation_from_es < 0:
                raise ValueError(
                    f"{ctx}.minimum_elevation_from_es needs to be a positive number!"
                )

        if any(cond not in self.__ALLOWED_CONDITIONS for cond in self.conditions):
            raise ValueError(
                f"{ctx}.conditions = {self.conditions}\n"
                f"However, only the following are allowed: {self.__ALLOWED_CONDITIONS}"
            )

        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError(
                f"{ctx}.conditions = {self.conditions}\n"
                "And it contains duplicate values!"
            )


@dataclass
class ParametersImtMssDc(ParametersBase):
    """Dataclass for the IMT MSS-DC topology parameters."""
    section_name: str = "imt_mss_dc"

    nested_parameters_enabled = True

    # MSS_D2D system name
    name: str = "SystemA"

    # Orbit parameters
    orbits: list[ParametersOrbit] = field(default_factory=lambda: [ParametersOrbit()])

    # Number of beams
    num_beams: int = 19

    # Beam radius in meters
    # The beam radius should be calculated based on the Antenna Pattern used for IMT Space Stations
    beam_radius: float = 36516.0

    sat_is_active_if: ParametersSelectActiveSatellite = field(default_factory=ParametersSelectActiveSatellite)

    def validate(self, ctx: str):
        """
        Raises
        ------
        ValueError
            If a parameter is not valid.
        """
        # Now do the sanity check for some parameters
        if self.num_beams not in [1, 7, 19]:
            raise ValueError(f"{ctx}.num_beams: Invalid number of sectors {self.num_sectors}")

        if self.beam_radius <= 0:
            raise ValueError(f"{ctx}.beam_radius: cell_radius must be greater than 0, but is {self.cell_radius}")
        else:
            self.cell_radius = self.beam_radius
            self.intersite_distance = np.sqrt(3) * self.cell_radius

        super().validate(ctx)

