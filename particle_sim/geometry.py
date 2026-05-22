import shapely
import numpy as np
from scipy.ndimage import distance_transform_edt
from jax.scipy.ndimage import map_coordinates
from shapely import Geometry