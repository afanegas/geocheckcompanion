"""
Geo-Check Companion - Geo-Check Functions

Author: Alejandro Fanegas
Version: 1.0.0 (2025-05-25)
License: MIT License

This application is a companion tool for the Geo-Check App, designed to calculate and analyze
geothermal potential using G-functions and heat extraction calculations. It provides a user-friendly
interface for importing borehole data, calculating G-functions, and performing detailed heat
extraction analysis according to VDI 4640 standards.

"""

import json
import pygfunction as gt
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import math
import cmath
import io

def geojson_to_boreholes(geojson_data, D=2, r_b=0.075):
    """
    Convert geothermal points from GeoJSON to pygfunction borehole objects.
    Coordinates are transformed to meters using the first point as reference (0,0).
    The depth property from each point is used as the borehole depth.
    
    Args:
        geojson_data (dict): GeoJSON data
        D (float): Borehole buried depth in meters (default: 2)
        r_b (float): Borehole radius in meters (default: 0.075)
        
    Returns:
        list: List of pygfunction borehole objects
    """
    # Extract geothermal points
    geothermal_points = geojson_data['geothermalPointsLayer']['features']
    
    # Get the reference point (first point)
    ref_coords = geothermal_points[0]['geometry']['coordinates']
    ref_lon, ref_lat = ref_coords[0], ref_coords[1]
    
    # Create transformer from WGS84 to a local projection centered at reference point
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
        always_xy=True
    )
    
    # Convert points to boreholes
    boreholes = []
    point_info = []
    
    for i, point in enumerate(geothermal_points, 1):
        # Extract coordinates
        coords = point['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        
        # Extract depth from properties
        depth = point['properties'].get('depth', 100)  # Default to 100m if depth not specified
        
        # Transform coordinates to meters
        x, y = transformer.transform(lon, lat)
        point_info.append({
            "index": i,
            "x": x,
            "y": y,
            "depth": depth
        })
        
        # Create borehole object with depth from properties
        # H: depth of borehole (m)
        # D: borehole buried depth (m)
        # r_b: borehole radius (m)
        borehole = gt.boreholes.Borehole(
            H=depth,  # Use depth from properties
            D=D,      # Use provided buried depth
            r_b=r_b,  # Use provided radius
            x=x,      # X coordinate in meters
            y=y       # Y coordinate in meters
        )
        boreholes.append(borehole)
    
    return boreholes, point_info

def calculate_g_function(borehole_field, tmax_years=3000, dt=3600, Nt=50, alpha=1.0e-6):
    """
    Calculate the g-function for a given borehole field.
    
    Args:
        borehole_field (list): List of pygfunction borehole objects
        tmax_years (float): Maximum simulation time in years
        dt (float): Initial time step in seconds
        Nt (int): Number of time steps
        alpha (float): Ground thermal diffusivity in m²/s
        
    Returns:
        tuple: (time, g_function, g_value_at_target, ts) where:
            - time is the time array
            - g_function is the calculated g-function
            - g_value_at_target is the g-function value at ln(t/ts)=2
            - ts is the characteristic time
    """
    # Calculate average depth for characteristic time
    avg_depth = sum(b.H for b in borehole_field) / len(borehole_field)
    ts = (avg_depth**2) / (9 * alpha)  # Characteristic time
    
    # Calculate time steps
    time = gt.utilities.time_geometric(dt=dt, tmax=tmax_years*8760*dt, Nt=Nt)
    
    # Calculate the exact time value for ln(t/ts)=2
    # ln(t/ts) = 2
    # t/ts = e^2
    # t = ts * e^2
    t_ln_2 = ts * np.exp(2)
    
    # Add the exact time value to the time array and sort
    time = np.append(time, t_ln_2)
    time.sort()
    
    # Calculate g-function
    g_function = gt.gfunction.uniform_temperature(borehole_field, time, alpha)
    
    # Find the g-value at ln(t/ts)=2
    ln_t_ts = np.log(time / ts)
    target_ln_t_ts = 2.0
    idx = np.where(np.isclose(ln_t_ts, target_ln_t_ts, rtol=1e-10, atol=1e-10))[0]
    
    # We should always have an exact match now
    g_value_at_target = g_function[idx[0]]
    
    return time, g_function, g_value_at_target, ts

def geohand_clone(T_surface, EWS_length, EWS_count, Lambda, q_geo, GVal, Usage=1800, r_b=0.075, R_b=0.1, dT_Sole=4, monthly_share=0.16):
    """
    Calculate geothermal parameters for a borehole field based on Geohand-Light and VDI 4640 guidelines.
    
    Args:
        T_surface (float): Surface temperature in °C
        EWS_length (float): Length of each borehole in meters
        EWS_count (int): Number of boreholes in the field
        Lambda (float): Thermal conductivity of the ground in W/mK
        q_geo (float): Geothermal heat flow in W/m²
        GVal (float): G-function value at ln(t/ts)=2
        Usage (float, optional): Annual operation hours. Defaults to 1800.
        r_b (float, optional): Borehole radius in meters. Defaults to 0.075.
        R_b (float, optional): Effective borehole thermal resistance in m*K/W. Defaults to 0.1.
        dT_Sole (float, optional): Maximum difference between inlet and outlet temperature in °C. Defaults to 4.
        monthly_share (float, optional): Maximaler Monatsanteil am jährlichen Wärmeentzug [%]. Defaults to 0.16.
        
    Returns:
        dict: Dictionary containing the following results:
            - q_ews_max: Maximum heat extraction rate in kW/m
            - P_EWS_max: Maximum power of the borehole field in kW
            - E_max: Maximum annual energy extraction in kWh
            - Grundlast: Base load in W/m
            - ZyklischeLast: Cyclic load in W/m
            - SpitzenLast: Peak load in W/m
            - T_in: Minimum inlet temperature in °C
            - T_per: Monthly average inlet temperature in °C
    """
    ### Fixed data ###
    q_ews_max = 0.1  # Start for the iteration heat extraction rate (kW/m)
      
    ### Internal parameters to calculate ####
    T_soil = T_surface + ((EWS_length * q_geo) / (2 * Lambda))  # Temperature of the soil
    rb_h = r_b / EWS_length
    g_corr = GVal
    R_stat = 1 / (2 * Lambda * math.pi) * g_corr

    Eindring = math.sqrt(Lambda / (2.18 * 1000000) * (8760 * 3600) / math.pi)
    R_cycl = 1 / (2 * math.pi * Lambda) * math.sqrt((math.log(2 / (r_b * math.sqrt(2) / Eindring)) - 0.5722)**2 + math.pi**2 / 16)

    R_max = 1 / (2 * math.pi * Lambda) * (math.log(math.sqrt(4 * Lambda / (2.18 * 1000000) * 24 * 3600) / r_b) - 0.5722 / 2)

    q_ews_max_1 = q_ews_max  # Initial value, will be iteratively adjusted

    # Function to calculate T_in for a given q_ews_max_1
    def fun_T_in(q_ews_max_1):
        return (T_soil + (
            (-(q_ews_max_1 * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760)) * (R_b + R_stat) +
            (-(((q_ews_max_1 * EWS_length * EWS_count * Usage) * monthly_share) * 1000 / (EWS_length * EWS_count * 730) -
              ((q_ews_max_1 * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760)))) * (R_cycl + R_b) +
            (-((EWS_length * EWS_count * q_ews_max_1) * 1000 / (EWS_length * EWS_count) - 
              ((q_ews_max_1 * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760)) -
              (((q_ews_max_1 * EWS_length * EWS_count * Usage) * monthly_share) * 1000 / (EWS_length * EWS_count * 730) - 
               ((q_ews_max_1 * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760))))) * (R_b + R_max))) - (dT_Sole / 2)

    # Initial calculation of T_in (for the max)
    T_in = fun_T_in(q_ews_max_1)

    #### As in VDI 4640 the inlet temperature should not fall -5degC ####
    while T_in < -5:
        q_ews_max_1 = q_ews_max_1 - 0.0001
        T_in = fun_T_in(q_ews_max_1)
    while T_in > -5.05:
        q_ews_max_1 = q_ews_max_1 + 0.0001
        T_in = fun_T_in(q_ews_max_1)
    
    # Function for calculating the monthly average inlet temperature
    def fun_T_per(q_ews_max_1):
      return (T_soil -(((q_ews_max_1 * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760))) * (R_b + R_stat)
              + -(((q_ews_max_1 * EWS_length * EWS_count * Usage * 1000) * monthly_share) / (730 * EWS_length * EWS_count) - 
                  ((q_ews_max_1 * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760))) * (R_cycl + R_b)
              - dT_Sole / 2)

    # Initial calculation of T_per
    T_per = fun_T_per(q_ews_max_1)

    # As in VDI 4640 the monthly average inlet temperature should not fall below 0degC
    while T_per < 0:
       q_ews_max_1 = q_ews_max_1 - 0.0001
       T_per = fun_T_per(q_ews_max_1)

    # Recalculate T_in with the adjusted q_ews_max_1
    T_in = fun_T_in(q_ews_max_1)

    # Calculate the three main cyclic components: Base load (Grundlast), cyclic load (ZyklischeLast) and peak load (SpitzenLast)
    Grundlast = ((q_ews_max_1 * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760))
    ZyklischeLast = (((q_ews_max_1 * EWS_length * EWS_count * Usage * 1000) * monthly_share) / (730 * EWS_length * EWS_count) - 
                    ((q_ews_max_1 * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760)))
    SpitzenLast = (EWS_length * EWS_count * q_ews_max_1 * 1000) / ((EWS_length * EWS_count)) - Grundlast - ZyklischeLast

    # Calculate temperature changes due to the loads
    dT_stat = -Grundlast * (R_b + R_stat)
    dT_cycl = -ZyklischeLast * (R_cycl + R_b)
    dT_max = -SpitzenLast * (R_b + R_max)
    
    # Return results
    return {
        "q_ews_max": q_ews_max_1,
        "P_EWS_max": q_ews_max_1 * EWS_count * EWS_length,  # Maximum power in kW
        "E_max": q_ews_max_1 * EWS_count * EWS_length * Usage,  # Maximum annual energy in kWh
        "Grundlast": Grundlast,
        "ZyklischeLast": ZyklischeLast,
        "SpitzenLast": SpitzenLast,
        "T_in": T_in,  # Minimal inlet temperature
        "T_per": T_per,  # Monthly average inlet temperature
        "T_soil": T_soil,
        "R_stat": R_stat,
        "R_cycl": R_cycl,
        "R_max": R_max,
        "dT_stat": dT_stat,
        "dT_cycl": dT_cycl,
        "dT_max": dT_max
    }

def geohand_clone_custom(T_surface, EWS_length, EWS_count, Lambda, q_geo, GVal, E_max, P_EWS_max, r_b=0.075, R_b=0.1, dT_Sole=4, monthly_share=0.16):
    """
    Custom version of geohand_clone. Calculate geothermal parameters for a borehole field based on Geohand-Light and VDI 4640 guidelines.
    Args:
        T_surface (float): Surface temperature in °C
        EWS_length (float): Length of each borehole in meters
        EWS_count (int): Number of boreholes in the field
        Lambda (float): Thermal conductivity of the ground in W/mK
        q_geo (float): Geothermal heat flow in W/m²
        GVal (float): G-function value at ln(t/ts)=2
        E_max (float): Maximum annual energy extraction in kWh
        P_EWS_max (float): Maximum power of the borehole field in kW
        r_b (float, optional): Borehole radius in meters. Defaults to 0.075.
        R_b (float, optional): Effective borehole thermal resistance in m*K/W. Defaults to 0.1.
        dT_Sole (float, optional): Maximum difference between inlet and outlet temperature in °C. Defaults to 4.
        monthly_share (float, optional): Maximaler Monatsanteil am jährlichen Wärmeentzug [%]. Defaults to 0.16.
    Returns:
        dict: Dictionary containing the following results:
            - q_ews_max: Maximum heat extraction rate in kW/m
            - P_EWS_max: Maximum power of the borehole field in kW
            - E_max: Maximum annual energy extraction in kWh
            - Grundlast: Base load in W/m
            - ZyklischeLast: Cyclic load in W/m
            - SpitzenLast: Peak load in W/m
            - T_in: Minimum inlet temperature in °C
            - T_per: Monthly average inlet temperature in °C
    """
    ### Calculate Usage and q_ews_max ###
    Usage = E_max / P_EWS_max  # Calculate Usage from E_max and P_EWS_max
    q_ews_max = P_EWS_max / (EWS_count * EWS_length)  # Calculate q_ews_max from P_EWS_max
      
    ### Internal parameters to calculate ####
    T_soil = T_surface + ((EWS_length * q_geo) / (2 * Lambda))  # Temperature of the soil
    rb_h = r_b / EWS_length
    g_corr = GVal
    R_stat = 1 / (2 * Lambda * math.pi) * g_corr

    Eindring = math.sqrt(Lambda / (2.18 * 1000000) * (8760 * 3600) / math.pi)
    R_cycl = 1 / (2 * math.pi * Lambda) * math.sqrt((math.log(2 / (r_b * math.sqrt(2) / Eindring)) - 0.5722)**2 + math.pi**2 / 16)

    R_max = 1 / (2 * math.pi * Lambda) * (math.log(math.sqrt(4 * Lambda / (2.18 * 1000000) * 24 * 3600) / r_b) - 0.5722 / 2)

    # Function to calculate T_in for a given q_ews_max
    def fun_T_in(q_ews_max):
        return (T_soil + (
            (-(q_ews_max * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760)) * (R_b + R_stat) +
            (-(((q_ews_max * EWS_length * EWS_count * Usage) * monthly_share) * 1000 / (EWS_length * EWS_count * 730) -
              ((q_ews_max * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760)))) * (R_cycl + R_b) +
            (-((EWS_length * EWS_count * q_ews_max) * 1000 / (EWS_length * EWS_count) - 
              ((q_ews_max * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760)) -
              (((q_ews_max * EWS_length * EWS_count * Usage) * monthly_share) * 1000 / (EWS_length * EWS_count * 730) - 
               ((q_ews_max * EWS_length * EWS_count * Usage) * 1000 / (EWS_length * EWS_count * 8760))))) * (R_b + R_max))) - (dT_Sole / 2)

    # Calculate T_in
    T_in = fun_T_in(q_ews_max)
    
    # Function for calculating the monthly average inlet temperature
    def fun_T_per(q_ews_max):
      return (T_soil -(((q_ews_max * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760))) * (R_b + R_stat)
              + -(((q_ews_max * EWS_length * EWS_count * Usage * 1000) * monthly_share) / (730 * EWS_length * EWS_count) - 
                  ((q_ews_max * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760))) * (R_cycl + R_b)
              - dT_Sole / 2)

    # Calculate T_per
    T_per = fun_T_per(q_ews_max)

    # Calculate the three main cyclic components: Base load (Grundlast), cyclic load (ZyklischeLast) and peak load (SpitzenLast)
    Grundlast = ((q_ews_max * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760))
    ZyklischeLast = (((q_ews_max * EWS_length * EWS_count * Usage * 1000) * monthly_share) / (730 * EWS_length * EWS_count) - 
                    ((q_ews_max * EWS_length * EWS_count * Usage * 1000) / (EWS_length * EWS_count * 8760)))
    SpitzenLast = (EWS_length * EWS_count * q_ews_max * 1000) / ((EWS_length * EWS_count)) - Grundlast - ZyklischeLast

    # Calculate temperature changes due to the loads
    dT_stat = -Grundlast * (R_b + R_stat)
    dT_cycl = -ZyklischeLast * (R_cycl + R_b)
    dT_max = -SpitzenLast * (R_b + R_max)
    
    # Return results
    return {
        "q_ews_max": q_ews_max,
        "P_EWS_max": P_EWS_max,  # Use input P_EWS_max
        "E_max": E_max,  # Use input E_max
        "Usage": Usage,  # Add Usage to return values
        "Grundlast": Grundlast,
        "ZyklischeLast": ZyklischeLast,
        "SpitzenLast": SpitzenLast,
        "T_in": T_in,  # Minimal inlet temperature
        "T_per": T_per,  # Monthly average inlet temperature
        "T_soil": T_soil,
        "R_stat": R_stat,
        "R_cycl": R_cycl,
        "R_max": R_max,
        "dT_stat": dT_stat,
        "dT_cycl": dT_cycl,
        "dT_max": dT_max
    }

def plot_borehole_field(field):
    """
    Plot the borehole field layout and return the figure as a bytes object.
    
    Args:
        field (list): List of pygfunction borehole objects
        
    Returns:
        bytes: The figure as a bytes object
    """
    # Set modern style for matplotlib
    plt.style.use('default')
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 8))
    
    # Plot the borehole field
    gt.boreholes.visualize_field(field)
    
    # Get the current axes and customize
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.set_title('Borehole Field Layout', color='black', fontsize=14, pad=15)
    ax.set_xlabel('X (m)', color='black')
    ax.set_ylabel('Y (m)', color='black')
    ax.tick_params(colors='black')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def plot_g_function(time, g_function, g_value_at_target, ts):
    """
    Plot the g-function for a borehole field and return the figure as a bytes object.
    
    Args:
        time (numpy.ndarray): Time array in seconds
        g_function (numpy.ndarray): Calculated g-function values
        g_value_at_target (float): G-function value at ln(t/ts)=2
        ts (float): Characteristic time in seconds
        
    Returns:
        bytes: The figure as a bytes object
    """
    # Set modern style for matplotlib
    plt.style.use('default')
    
    # Create figure with modern styling
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    
    # Calculate ln(t/ts)
    ln_t_ts = np.log(time / ts)
    
    # Plot the g-function with modern styling
    ax.plot(ln_t_ts, g_function, color='#4a9eff', linewidth=2)
    
    # Mark the point on the plot
    ax.plot(2.0, g_value_at_target, 'ro', markersize=8, color='#ff6b6b')
    ax.annotate(f'(2.0, {g_value_at_target:.4f})', 
                (2.0, g_value_at_target), 
                xytext=(10, 10), textcoords='offset points',
                color='black', fontsize=10, bbox=dict(facecolor='white', edgecolor='#4a9eff', alpha=0.7))
    
    # Customize the plot
    ax.set_xlabel('ln(t/ts)', color='black', fontsize=12)
    ax.set_ylabel('g-function', color='black', fontsize=12)
    ax.set_title('G-function for the borehole field', color='black', fontsize=14, pad=15)
    ax.tick_params(colors='black')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf 
