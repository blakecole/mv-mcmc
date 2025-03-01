# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: geotools.py                                       #
#    DATE: 28 FEB 2025                                       #
# ********************************************************** #

import re
import csv
import math


def read_kml_points(kml_path):
    """
    Reads a list of coordinates in KML format, stores in list of tuples.
    
    Args:
        kml_path (string): Path to KML file.
    
    Returns:
        data: A list of coordinate points, as (lon lat) tuples.
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
    data = []
    coord_entries = coords_text.strip().split()  # Splits on any whitespace
    for entry in coord_entries:
        # Each entry should be in the format: lon,lat,elevation
        parts = entry.split(',')
        if len(parts) >= 3:
            lon = float(parts[0])
            lat = float(parts[1])
            elev = float(parts[2])
            data.append((lon, lat))
        else:
            print("Skipping entry (unexpected format):", entry)

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
    read_kml_points(kml_path)
        
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
    x = lon * math.cos(math.radians(avg_lat))
    y = lat
    return (x, y)


def is_point_in_region(point, vertices):
    """
    Determines if a point (lat, lon) is inside a polygon defined by a list of 
    (lat, lon) points, using a simple transformation to account for Earth's 
    curvature by scaling longitudes.
    
    Args:
        point (tuple): The point to check, as (lat, lon).
        vertices (list): A list of (lat, lon) tuples defining the polygon
                         vertices.  The polygon should be closed (i.e. first
                         and last points are the same, or nearly the same)
    
    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    # Compute the average latitude of the polygon vertices for scaling
    avg_lat = sum(lat for lat, lon in vertices) / len(vertices)
    
    # Transform the point and polygon vertices
    transformed_point = equirectangular(point, avg_lat)
    transformed_vertices = [equirectangular(v, avg_lat) for v in vertices]
    
    x, y = transformed_point
    inside = False
    n = len(transformed_vertices)
    
    # Ray-casting algorithm on the transformed coordinates
    for i in range(n):
        j = (i - 1) % n
        xi, yi = transformed_vertices[i]
        xj, yj = transformed_vertices[j]
        
        # Check if the ray crosses the edge between vertices i and j
        if ((yi > y) != (yj > y)):
            intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < intersect:
                inside = not inside
    
    return inside
