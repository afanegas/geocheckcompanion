import streamlit as st
import json
import pygfunction as gt
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import math
import cmath
import os
from PIL import Image
import re
import io
from geo_check_functions import geojson_to_boreholes, calculate_g_function, geohand_clone, plot_borehole_field, plot_g_function

# Set page configuration
st.set_page_config(
    page_title="Geo-Check Companion",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Default values for parameters
default_values = {
    # Borehole parameters
    "D": 2.0,
    "r_b": 0.075,
    "alpha": 1.0e-6,
    "tmax_years": 3000,
    "dt": 3600,
    "Nt": 50,
    
    # Heat extraction parameters
    "T_surface": 11.0,
    "Lambda": 2.5,
    "q_geo": 0.065,
    "Usage": 1800,
    "R_b": 0.1,
    "dT_Sole": 4.0,
    "monthly_share": 16.0
}

# Initialize session state variables
if 'field' not in st.session_state:
    st.session_state.field = None
if 'time' not in st.session_state:
    st.session_state.time = None
if 'g_function' not in st.session_state:
    st.session_state.g_function = None
if 'g_value_at_target' not in st.session_state:
    st.session_state.g_value_at_target = None
if 'ts' not in st.session_state:
    st.session_state.ts = None
if 'geojson_data' not in st.session_state:
    st.session_state.geojson_data = None
if 'point_info' not in st.session_state:
    st.session_state.point_info = None
if 'borehole_field_plot' not in st.session_state:
    st.session_state.borehole_field_plot = None
if 'g_function_plot' not in st.session_state:
    st.session_state.g_function_plot = None
if 'reset_g_function' not in st.session_state:
    st.session_state.reset_g_function = False
if 'reset_heat_extraction' not in st.session_state:
    st.session_state.reset_heat_extraction = False

# Initialize input parameters in session state
for key in default_values:
    if key not in st.session_state:
        st.session_state[key] = default_values[key]

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0066cc;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #0066cc;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #333333;
    }
    .result-text {
        font-size: 1.1rem;
        font-weight: bold;
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>Geo-Check Companion</h1>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Erdwärmesonden & G-Funktion", "Wärmeentzugsanalyse", "Hilfe & Informationen"])

# Tab 1: Borehole and G-Function
with tab1:
    # File upload section first
    st.markdown("<h3 class='subsection-header'>Erdwärmesonden importieren</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Wählen Sie eine GeoJSON-Datei aus", type=["geojson"])
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            geojson_data = json.load(uploaded_file)
            st.session_state.geojson_data = geojson_data
            
            # Display success message
            st.success(f"Datei erfolgreich importiert: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei: {str(e)}")
    
    # Parameters section with expandable sections
    st.markdown("<h3 class='subsection-header'>Parameter</h3>", unsafe_allow_html=True)
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Borehole parameters
        with st.expander("Erdwärmesonden-Parameter", expanded=True):
            st.number_input("Überdeckungshöhe (m)", value=st.session_state.D, min_value=0.0, step=0.1, 
                                help="Tiefe von der Oberfläche bis zum Sondenkopf in Metern", key="D")
            st.number_input("Bohrlochradius (m)", value=st.session_state.r_b, min_value=0.01, step=0.005, 
                                help="Radius des Bohrlochs in Metern", key="r_b")
            st.number_input("Temperaturleitfähigkeit Erdreich (m²/s)", value=st.session_state.alpha, 
                                format="%.2e", help="Temperaturleitfähigkeit des Untergrunds", key="alpha")
    
    with col2:
        # G-Function parameters
        with st.expander("Darstellung der G-Funktion", expanded=False):
            st.number_input("Max. Simulationszeit (Jahre)", value=st.session_state.tmax_years, 
                                    min_value=1, step=100, help="Maximale Simulationszeit in Jahren", key="tmax_years")
            st.number_input("Anfangs-Zeitschritt (s)", value=st.session_state.dt, 
                            min_value=100, step=100, help="Anfangs-Zeitschritt in Sekunden", key="dt")
            st.number_input("Anzahl Zeitschritte", value=st.session_state.Nt, 
                            min_value=10, step=10, help="Anzahl der Zeitschritte für die Simulation", key="Nt")
    
    # Reset button
    if st.button("Standardwerte wiederherstellen", key="reset_g_function_button"):
        # Create a container for the reset message
        reset_container = st.empty()
        reset_container.success("Parameter werden zurückgesetzt...")
        
        # Use a callback to reset the values
        def reset_g_function_values():
            for key in ["D", "r_b", "alpha", "tmax_years", "dt", "Nt"]:
                if key in st.session_state:
                    del st.session_state[key]
        
        reset_g_function_values()
        reset_container.success("Parameter wurden auf Standardwerte zurückgesetzt!")
        st.rerun()
    
    # G-Function calculation section
    st.markdown("<h3 class='subsection-header'>Erwärmesonden verarbeiten</h3>", unsafe_allow_html=True)
    if st.button("Erwärmesonden verarbeiten und G-Funktion berechnen"):
        if st.session_state.geojson_data is None:
            st.error("Bitte importieren Sie zuerst eine GeoJSON-Datei.")
        else:
            with st.spinner("G-Funktion wird berechnet..."):
                try:
                    # Convert GeoJSON to boreholes
                    field, point_info = geojson_to_boreholes(
                        st.session_state.geojson_data, 
                        D=st.session_state.D, 
                        r_b=st.session_state.r_b
                    )
                    st.session_state.field = field
                    st.session_state.point_info = point_info
                    
                    # Calculate the g-function
                    time, g_function, g_value_at_target, ts = calculate_g_function(
                        field,
                        tmax_years=st.session_state.tmax_years,
                        dt=st.session_state.dt,
                        Nt=st.session_state.Nt,
                        alpha=st.session_state.alpha
                    )
                    st.session_state.time = time
                    st.session_state.g_function = g_function
                    st.session_state.g_value_at_target = g_value_at_target
                    st.session_state.ts = ts
                    
                    # Generate plots
                    borehole_field_plot = plot_borehole_field(field)
                    g_function_plot = plot_g_function(time, g_function, g_value_at_target, ts)
                    st.session_state.borehole_field_plot = borehole_field_plot
                    st.session_state.g_function_plot = g_function_plot
                    
                    # Display success message
                    st.success("G-Funktion erfolgreich berechnet! Sie können weiter zum Reiter '**Wärmeentzungsanalyse**' gehen.")
                    
                    # Display results
                    st.markdown("<h4>Berechnungsergebnisse:</h4>", unsafe_allow_html=True)
                    
                    # Calculate average depth
                    avg_depth = sum(b.H for b in field) / len(field)
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<p class='result-text'>Erdwärmesondenfeld Information:</p>", unsafe_allow_html=True)
                        st.write(f"Anzahl der Erdwärmesonden: {len(field)}")
                        st.write(f"Durchschnittliche Tiefe: {avg_depth:.2f} m")
                        st.write(f"G-Funktionswert bei ln(t/ts) = 2: {g_value_at_target:.4f}")
                    
                    with col2:
                        st.markdown("<p class='result-text'>Charakteristische Zeit:</p>", unsafe_allow_html=True)
                        st.write(f"ts = {ts:.2e} Sekunden")
                        st.write(f"ts = {ts/(365*24*3600):.2f} Jahre")
                    
                    # Display plots with reduced size (60%)
                    st.markdown("<h4>Erdwärmesondenfeld Layout:</h4>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(st.session_state.borehole_field_plot, use_container_width=True)
                    
                    st.markdown("<h4>G-Funktion:</h4>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(st.session_state.g_function_plot, use_container_width=True)
                    
                    # Display point information
                    with st.expander("Erdwärmesonden Details anzeigen"):
                        st.markdown("<h4>Erdwärmesonden Koordinaten und Tiefen:</h4>", unsafe_allow_html=True)
                        st.dataframe(point_info)
                    
                except Exception as e:
                    st.error(f"Fehler bei der Berechnung der G-Funktion: {str(e)}")

# Tab 2: Heat Extraction Analysis
with tab2:
    # Heat extraction parameters
    st.markdown("<h3 class='subsection-header'>Wärmeentzugs-Parameter</h3>", unsafe_allow_html=True)
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Grundparameter", expanded=True):
            st.number_input("Wärmeleitfähigkeit (W/mK)", value=st.session_state.Lambda, 
                                min_value=0.1, step=0.1, help="Wärmeleitfähigkeit des Erdreichs", key="Lambda")
            st.number_input("Vollaststunden Heizung (h)", value=st.session_state.Usage, 
                                min_value=100, step=100, 
                                help="Äquivalente Anzahl der Stunden, die die Anlage mit Nennleistung im Heizbetrieb arbeitet", key="Usage")
            st.number_input("Max. Monatsanteil am jährlichen Wärmeentzug (%)", 
                                    value=st.session_state.monthly_share, min_value=1.0, max_value=100.0, step=1.0, 
                                    help="Prozentualer Anteil des Monats mit der größten Entzugsmenge am gesamten jährlichen Wärmeentzug [%] (Standard: 16%)", key="monthly_share")
    
    with col2:
        with st.expander("Zusätzliche Parameter", expanded=False):
            st.number_input("Oberflächentemperatur (°C)", value=st.session_state.T_surface, 
                                    min_value=-10.0, max_value=30.0, step=0.1, 
                                    help="Oberflächentemperatur in Celsius", key="T_surface")
            st.number_input("Geothermischer Wärmestromdichte (W/m²)", value=st.session_state.q_geo, 
                                min_value=0.01, step=0.01, help="Geothermischer Wärmestromdichte in W/m²", key="q_geo")
            st.number_input("Bohrlochwiderstand (m*K/W)", value=st.session_state.R_b, 
                            min_value=0.01, step=0.01, help="Bohrlochwiderstand", key="R_b")
            st.number_input("Max. Temperaturdifferenz Wärmepumpenaustritt/Eintritt (°C)", 
                                value=st.session_state.dT_Sole, min_value=1.0, step=0.5, 
                                help="Größter Unterschied zwischen der Temperatur der Sole beim Austritt aus und beim Eintritt in die Wärmepumpe", key="dT_Sole")
    
    # Reset button
    if st.button("Standardwerte wiederherstellen", key="reset_heat_extraction_button"):
        # Create a container for the reset message
        reset_container = st.empty()
        reset_container.success("Parameter werden zurückgesetzt...")
        
        # Use a callback to reset the values
        def reset_heat_extraction_values():
            for key in ["Lambda", "Usage", "monthly_share", "T_surface", "q_geo", "R_b", "dT_Sole"]:
                if key in st.session_state:
                    del st.session_state[key]
        
        reset_heat_extraction_values()
        reset_container.success("Parameter wurden auf Standardwerte zurückgesetzt!")
        st.rerun()
    
    # Heat extraction analysis section
    st.markdown("<h3 class='subsection-header'>Wärmeentzugsanalyse</h3>", unsafe_allow_html=True)
    
    if st.button("Wärmeentzug berechnen"):
        if st.session_state.g_value_at_target is None:
            st.error("Bitte berechnen Sie zuerst die G-Funktion.")
        else:
            with st.spinner("Wärmeentzug wird berechnet..."):
                try:
                    # Calculate EWS_length (average depth) and EWS_count from the field
                    EWS_length = sum(b.H for b in st.session_state.field) / len(st.session_state.field)  # Average depth of boreholes
                    EWS_count = len(st.session_state.field)  # Number of boreholes
                    
                    # Call the geohand_clone function with the defined parameters
                    results = geohand_clone(
                        T_surface=st.session_state.T_surface,
                        EWS_length=EWS_length,
                        EWS_count=EWS_count,
                        Lambda=st.session_state.Lambda,
                        q_geo=st.session_state.q_geo,
                        GVal=st.session_state.g_value_at_target,
                        Usage=st.session_state.Usage,
                        r_b=st.session_state.r_b,
                        R_b=st.session_state.R_b,
                        dT_Sole=st.session_state.dT_Sole,
                        monthly_share=st.session_state.monthly_share / 100  # Convert percentage to decimal
                    )
                    
                    # Display success message
                    st.success("Wärmeentzugsanalyse erfolgreich durchgeführt!")
                    
                    # Display results
                    st.markdown("<h4>Wärmeentzugs-Ergebnisse:</h4>", unsafe_allow_html=True)
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        #st.markdown("<p class='result-text'>Maximale Entzugsleistung:</p>", unsafe_allow_html=True)
                        st.write(f"Maximale spezifische Entzugsleistung: {results['q_ews_max'] * 1000:.2f} W/m")
                        st.write(f"Maximale Entzugsleistung: {results['P_EWS_max']:.2f} kW")
                        st.write(f"Maximaler jährlicher Wärmeentzug: {results['E_max']:.2f} kWh")
                    
                    # Display calculation conditions
                    #st.markdown("<h4>Berechnungsbedingungen:</h4>", unsafe_allow_html=True)
                    #st.write("- Basierend auf Geohand-Light und VDI 4640 Richtlinien für die Auslegung von Erdwärmesondenfeldern")
                    #st.write("- Die Vorlauftemperatur darf nicht unter -5°C fallen")
                    #st.write("- Die monatliche Durchschnittsvorlauftemperatur darf nicht unter 0°C fallen")
                    
                except Exception as e:
                    st.error(f"Fehler bei der Wärmeentzugsanalyse: {str(e)}")

# Tab 3: Help section
with tab3:
    st.markdown("""
    ## Geo-Check Companion Hilfe
    
    Diese Anwendung berechnet den maximal möglichen Wärmeentzug unter Einhaltung der Temperaturgrenzen gemäß VDI 4640. 
    Die Berechnung basiert auf der Methodik der vereinfachten Lastzerlegung, wie sie auch in GEO-HANDlight angewendet wird.
    Der Geo-Check Companion erweitert die Geo-Check-App und ermöglicht eine präzisere Analyse des Wärmeentzugs – insbesondere durch die genauere Berücksichtigung der gegenseitigen thermischen Beeinflussung benachbarter Erdwärmesonden mithilfe der G-Funktionsberechnung. Für eine gute Zusammenfassung der gesamten Methodik s. Referenz 1 - '2.3.2. Calculation of the technical geothermal potential and heat supply rate.'
    
    ### Wie Sie dieses Tool verwenden
    
    1. **Erdwärmesonden importieren**:
       - Wählen Sie eine GeoJSON-Datei mit geothermischen Punkten aus
       - Die GeoJSON-Datei sollte mit der Geo-Check-Anwendung erstellt worden sein
    
    2. **G-Funktion berechnen**:
       - Klicken Sie auf "G-Funktion berechnen"
       - Die G-Funktion wird berechnet und visualisiert
    
    3. **Wärmeentzugsanalyse**:
       - Klicken Sie auf "Wärmeentzug berechnen"
       - Die Wärmeentzugsanalyse wird durchgeführt und die Ergebnisse werden angezeigt
    
    ### Berechnungsgrundlagen
    
    Die Berechnungen in dieser Anwendung basieren auf folgenden Grundlagen:
    
    1. **G-Funktion Berechnung**:
       - Die G-Funktion wird mit der pygfunction-Library berechnet
       - Die Berechnung basiert auf der uniformen Temperaturrandbedingung (uniform temperature boundary condition)
       - Der G-Funktionswert bei ln(t/ts)=2 wird für die Wärmeentzugsanalyse verwendet (ca. Endwert)
    
    2. **Wärmeentzugsanalyse**:
       - Basierend auf GEO-HANDight und die VDI 4640 Richtlinie für die Auslegung von Erdwärmesondenfeldern
       - Gemäß VDI 4640 werden zwei Temperaturgrenzen berücksichtigt:
                
         • Im Heizbetrieb soll die Eintrittstemperatur des Wärmeträgermediums in die Erdwärmesonde(n) im Monatsmittel 0 °C nicht unterschreiten.
        
         • Bei Spitzenlast soll diese Temperatur –5 °C nicht unterschreiten.
       - Die Berechnung basierend auf GEO-HANDlight berücksichtigt drei Hauptkomponenten (Lastzerlegung):
         
         • Grundlast: Konstanter Wärmeentzug über das Jahr
        
         • Periodische Last: Monatliche Schwankungen im Wärmeentzug
        
         • Spitzenlast: Kurzzeitiger Spitzenwärmeentzug
    
    ### Eingabeparameter
    
    **Erdwärmesonden-Parameter:**
    - Überdeckungshöhe (D): Tiefe von der Oberfläche bis zum Sondenkopf in Metern
    - Bohrlochradius (r_b): Radius des Bohrlochs in Metern
    - Temperaturleitfähigkeit Erdreich (alpha): Temperaturleitfähigkeit des Untergrunds in m²/s
    
    **G-Funktion Parameter:**
    Betreffen nur die Darstellung der G-Funktion
    - Max. Simulationszeit: Maximale Simulationszeit in Jahren
    - Anfangs-Zeitschritt in Sekunden
    - Anzahl der Zeitschritte für die Simulation
    
    **Wärmeentzugs-Parameter:**
    - Oberflächentemperatur (ϑ_surf) in °C
    - Wärmeleitfähigkeit des Untergrunds (λ_E) in W/mK
    - Geothermischer Wärmestromdichte (q_geo) in W/m²
    - Vollaststunden Heizung: Äquivalente Anzahl der Stunden, die die Anlage mit Nennleistung im Heizbetrieb arbeitet
    - Sondenwiderstand (R_b) in m*K/W
    - Max. Temperaturdifferenz Wärmepumpenaustritt/Eintritt in °C
    - Monatsanteil: Maximaler Monatsanteil am jährlichen Wärmeentzug [%] (Standard: 16%)
    
    ### Ausgabeparameter
    
    - Maximale spezifische Entzugsleistung: Maximale Wärmeentzugsrate pro Meter Erdwärmesonde in W/m (spez. q_EWS_H)
    - Maximale Entzugsleistung: Maximale Leistung des Sondenfelds in kW (q_EWS_H)
    - Maximaler jährlicher Wärmeentzug in kWh
    
    ### Referenzen
    
    Die Basen für die Berechnungen in der Geo-Check-Begleiter können in den folgenden Papieren gelesen werden:
    
    1. Cimmino, M., & Bernier, M. (2021). A new approach to the calculation of the g-function for geothermal borehole fields. Applied Energy, 283, 116344. https://www.sciencedirect.com/science/article/pii/S096014812101822X
    
    2. Hochschule Biberach. (2022). GEO-HANDlight Version 5.0 – Benutzeranleitung (DocV 5.0). Hochschule Biberach. https://innosued.de/energie/geothermie-software-2/
    
    3. Cimmino, M., & Cook, J.C. (2022). pygfunction 2.2: https://pygfunction.readthedocs.io/en/stable/modules/gfunction.html
    """) 