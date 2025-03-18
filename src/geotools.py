# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: geotools.py                                       #
#    DATE: 28 FEB 2025                                       #
# ********************************************************** #

import re
import csv
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

# Earth radius in meters (spherical approximation)
R = 6371000

def read_kml_points(kml_path):
    """
    Reads a list of coordinates in KML format, stores in ndarray.
    
    Args:
        kml_path (string): Path to KML file.
    
    Returns:
        data: A 2D array of coordinate points, as (lon lat) rows.
    """

    # Read the entire file content
    with open(kml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use reg exp to extract text between <coordinates> and </coordinates>
    match = re.search(r"<coordinates>(.*?)</coordinates>", content, re.DOTALL)
    if match:
        coords_text = match.group(1).strip()
    else:
        raise ValueError("No <coordinates> block found.")

    # Process each set of coordinates
    coord_entries = coords_text.strip().split()  # Splits on any whitespace
    n_entries = len(coord_entries)
    data = np.zeros((n_entries, 2), dtype=float)
    for i in range(n_entries):
        # Each entry should be in the format: lon,lat,elevation
        parts = coord_entries[i].split(',')
        if len(parts) >= 3:
            lon = float(parts[0])
            lat = float(parts[1])
            elev = float(parts[2])
            data[i,:] = [lon, lat]
        else:
            print("Skipping entry (unexpected format):", coord_entries[i])
    return data


def kml_to_csv(kml_path):
    """
    Converts a list of coordinates in KML format to CSV format.
    
    Args:
        kml_path (string): Path to KML file.
    
    Returns:
        Nothing ():
    """

    # Read KML file
    data = read_kml_points(kml_path)
        
    # Write the extracted data to a CSV file
    csv_path = kml_path[:-4] + '.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header row (optional)
        csvwriter.writerow(['Longitude', 'Latitude', 'Elevation'])
        # Write the coordinate data rows
        csvwriter.writerows(data)
        print("KML coordinate pairs successfully saved in CSV format.")
    return


def make_flowfield_interp_fns(lon, lat, t, u, v, interp_style="nearest"):
    """
    Creates interpolation functions for the lateral and longitudinal components
    of the current velocity field.

    Parameters:
      lon, lat: 1D arrays of spatial coordinates (degrees) of length N.
      t: 1D array of time values (length M).
      u, v: 2D arrays of velocity components with shape (N, M).
      interp_style: Either "nearest" or "linear".

    Returns:
      (u_interp_fn, v_interp_fn): A tuple of interpolation functions.
    """
    # Create a 2D grid of coordinates using meshgrid.
    # This creates arrays of shape (N, M) for lon, lat, and t.
    # Note: use indexing='ij' so that the first axis corresponds to lon/lat.
    LON, T = np.meshgrid(lon, t, indexing='ij')
    LAT, _   = np.meshgrid(lat, t, indexing='ij')
    
    # Flatten the coordinate and velocity arrays.
    lon_flat = LON.ravel()
    lat_flat = LAT.ravel()
    t_flat   = T.ravel()
    u_flat   = u.ravel()
    v_flat   = v.ravel()
    
    # Stack the coordinate arrays into a single 2D array of shape (N*M, 3).
    coords = np.column_stack((lon_flat, lat_flat, t_flat))
    
    # Choose the interpolator based on the interpolation style.
    if interp_style == "linear":
        u_interp_fn = LinearNDInterpolator(coords, u_flat)
        v_interp_fn = LinearNDInterpolator(coords, v_flat)
    elif interp_style == "nearest":
        u_interp_fn = NearestNDInterpolator(coords, u_flat)
        v_interp_fn = NearestNDInterpolator(coords, v_flat)
    else:
        raise ValueError("Unrecognized interp_style. Use 'linear' or 'nearest'.")
    
    return u_interp_fn, v_interp_fn


def equirectangular(point, avg_lat):
    """
    Projects a geodetic point (lat, lon) onto local planar coordinate frame,
    using a simple equirectangular (or "plate carrée") approximation.
    
    Args:
        point (tuple): A geodetic point, as (lat, lon).
        avg_lat (scalar): Average latitude of the planar coordinate frame.
    
    Returns:
        (x, y): A tuple containing the projected point
    """
    lat, lon = point
    # Scale longitude by the cosine of the average latitude
    x = lon * np.cos(np.radians(avg_lat))
    y = lat
    return np.array([x, y])


def is_point(pt):
    """
    Helper function to check if the input is a single (lon, lat) point.
    Accepts lists, tuples, or NumPy arrays with shape (2,).
    """
    if isinstance(pt, (list, tuple)):
        return (len(pt) == 2) and all(isinstance(x, (int, float)) for x in pt)
    
    # Accept NumPy arrays with exactly 2 elements.
    elif isinstance(pt, np.ndarray):
        return (pt.size == 2)
    
    return False


def haversine_distance(p1, p2):
    """
    Vectorized haversine distance.
    
    Parameters:
      p1, p2: Either a single point (lon, lat) or an array of points with shape (n, 2).
              Coordinates are in degrees.
              
    Returns:
      Distance in meters. If inputs are single points, returns a scalar.
      Otherwise, returns an array of distances.
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    
    lat1 = np.deg2rad(p1[:,1])
    lon1 = np.deg2rad(p1[:,0])
    lat2 = np.deg2rad(p2[:,1])
    lon2 = np.deg2rad(p2[:,0])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    # If originally scalar, return scalar
    if d.size == 1:
        return d.item()
    return d


def haversine_bearing(p1, p2):
    """
    Vectorized initial bearing (forward azimuth) from p1 to p2.
    
    Parameters:
      p1, p2: Either a single point (lon, lat) or an array of points with shape (n, 2).
              Coordinates are in degrees.
              
    Returns:
      Bearing in radians measured clockwise from true north.
      If inputs are single points, returns a scalar; otherwise, an array.
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    
    lat1 = np.deg2rad(p1[:,1])
    lon1 = np.deg2rad(p1[:,0])
    lat2 = np.deg2rad(p2[:,1])
    lon2 = np.deg2rad(p2[:,0])
    
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = np.arctan2(x, y) % (2*np.pi)
    if bearing.size == 1:
        return bearing.item()
    return bearing


def reckon(p, bearing, distance):
    """
    Vectorized destination point calculation.
    
    Given a starting point p (lon, lat) in degrees, bearing (radians) and a distance in meters,
    compute the destination point(s) along a great-circle.
    
    Parameters:
      p: Single point (lon, lat) or an array of points with shape (n,2).
      bearing: Scalar or an array of bearings (degrees) with shape (n,).
      distance: Scalar or an array of distances (meters) with shape (n,).
    
    Returns:
      Destination point(s) as an array of shape (n, 2) in degrees.
      If the inputs were scalars, a single tuple (lon, lat) is returned.
    """
    p = np.atleast_2d(p)
    bearing = np.atleast_1d(np.radians(bearing))
    distance = np.atleast_1d(distance)
    
    lat1 = np.deg2rad(p[:,1])
    lon1 = np.deg2rad(p[:,0])
    delta = distance / R  # angular distance in radians
    
    lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) + np.cos(lat1)*np.sin(delta)*np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat1),
                             np.cos(delta) - np.sin(lat1)*np.sin(lat2))
    lon2 = np.rad2deg(lon2)
    lat2 = np.rad2deg(lat2)
    
    dest = np.column_stack((lon2, lat2))
    if dest.shape[0] == 1:
        return (dest[0,0], dest[0,1])
    return dest


def is_point_in_region(points, polygon):
    """
    Vectorized point-in-polygon test using the ray-casting algorithm.
    
    Parameters:
      points: A single point (lon, lat) as a length-2 array/tuple or an array of shape (n, 2).
      polygon: An array of polygon vertices (lon, lat) with shape (m, 2). 
               The polygon is assumed to be closed (if not, it is closed automatically).
    
    Returns:
      If points is a single point, returns a boolean.
      If points is an array of points, returns a boolean array of shape (n,).
    """
    polygon = np.asarray(polygon)
    # Ensure the polygon is closed.
    if not np.array_equal(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
    
    points = np.atleast_2d(points)
    
    x = points[:,0][:, np.newaxis]  # shape (n_points, 1)
    y = points[:,1][:, np.newaxis]
    
    # Polygon vertices
    poly = polygon
    x_poly = poly[:,0]
    y_poly = poly[:,1]
    
    # For each edge in the polygon, check if the ray intersects the edge.
    x0 = x_poly[:-1]
    y0 = y_poly[:-1]
    x1 = x_poly[1:]
    y1 = y_poly[1:]
    
    # Test if y coordinate is between the endpoints of the edge.
    cond = ((y0 > y) != (y1 > y))
    # Compute intersection x coordinate for each edge.
    x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
    
    intersections = cond & (x < x_int)
    inside = np.sum(intersections, axis=1) % 2 == 1
    if inside.size == 1:
        return inside.item()
    return inside


def closest_appropach_to_region(point, polygon):
    """
    Compute the closest distance (in meters) and the bearing (in radians)
    from a geodetic point (lon, lat) to a closed polygon defined as a 2D NumPy array (N x 2)
    of (lon, lat) points, using vectorized geodetic functions.
    
    Uses:
      - haversine_distance: vectorized great-circle distance (inputs in degrees)
      - haversine_bearing: vectorized initial bearing (inputs in degrees; returns degrees)
      - reckon: vectorized destination point given start point, bearing, and distance (inputs in degrees)
      - is_point_in_region: vectorized point-in-polygon test
      
    Parameters:
      point: (lon, lat) in degrees.
      polygon: a NumPy array of shape (N, 2) with vertices in (lon, lat) order.
               The polygon is assumed to be in degrees.
               
    Returns:
      (best_distance, best_bearing) where:
         best_distance is the minimal distance in meters from the point to the polygon,
         best_bearing is the initial bearing (in degrees, clockwise from true north)
         from the point to the closest point on the polygon.
         
      If the point is inside the polygon, returns (0.0, None).
    """
    # Ensure polygon is a NumPy array.
    polygon = np.asarray(polygon)
    
    # Ensure the polygon is closed.
    if not np.array_equal(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
        
    # If the point is inside the polygon, return (0.0, None).
    if is_point_in_region(point, polygon):
        return 0.0, None
    
    # Separate polygon vertices into segments: A = polygon[:-1], B = polygon[1:].
    A = polygon[:-1]  # shape (M,2)
    B = polygon[1:]
    
    # Compute distances:
    # d13: distance from each vertex A to the point.
    d13 = haversine_distance(A, point)           # returns an array of shape (M,)
    # d12: distance along each segment from A to B.
    d12 = haversine_distance(A, B)                 # array of shape (M,)
    # d_BP: distance from each vertex B to the point.
    dBP = haversine_distance(B, point)             # array of shape (M,)
    
    # Compute bearings:
    # θ₁₂: bearing from A to B.
    theta12 = haversine_bearing(A, B)              # array of shape (M,)
    # θ₁₃: bearing from A to point.
    theta13 = haversine_bearing(A, point)          # array of shape (M,)
    
    # Angular distance from A to point (in radians)
    delta13 = d13 / R  # using Earth radius R = 6371000 m
    
    # Cross-track distance:
    # d_xt = |asin(sin(δ₁₃) * sin(θ₁₃ - θ₁₂))| * R
    d_xt = np.abs(np.arcsin(np.sin(delta13) * np.sin(theta13 - theta12))) * R
    
    # Along-track distance:
    # d_at = arccos(cos(δ₁₃) / cos(d_xt/R)) * R, with clamping to avoid domain errors.
    cos_dxt = np.cos(d_xt / R)
    cos_term = np.where(cos_dxt == 0, 1, np.cos(delta13) / cos_dxt)
    cos_term = np.clip(cos_term, -1, 1)
    d_at = np.arccos(cos_term) * R
    
    # For each segment, if d_at > d12 then the candidate distance is the closer of d13 and d_BP.
    # Otherwise, the candidate distance is d_xt.
    candidate_distance = np.where(d_at > d12, np.minimum(d13, dBP), d_xt)
    
    # Determine candidate point for each segment.
    # We'll build a list of candidate points (each as a NumPy array [lon, lat] in degrees).
    candidate_points = []
    M = A.shape[0]
    for i in range(M):
        if d_at[i] > d12[i]:
            # Projection falls outside the segment.
            if d13[i] <= dBP[i]:
                candidate_points.append(A[i])
            else:
                candidate_points.append(B[i])
        else:
            # Projection falls on the segment.
            # Use reckon: from A[i], along bearing theta12[i], for distance d_at[i].
            cp = reckon(A[i], theta12[i], d_at[i])
            # Convert the tuple to an array.
            candidate_points.append(np.array(cp))
    candidate_points = np.array(candidate_points)  # shape (M, 2)
    
    # Compute the bearing from the input point to each candidate point.
    # To use the vectorized haversine_bearing, create an array with the point repeated M times.
    repeated_point = np.repeat(np.atleast_2d(point), M, axis=0)
    candidate_bearings = haversine_bearing(repeated_point, candidate_points)  # array of shape (M,)
    
    # Choose the candidate with the minimal distance.
    idx = np.argmin(candidate_distance)
    best_distance = candidate_distance[idx]
    best_bearing = np.degrees(candidate_bearings[idx])
    
    return best_distance, best_bearing