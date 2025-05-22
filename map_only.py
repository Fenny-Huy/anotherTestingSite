import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import datetime

with open("results.json") as f:
    results = json.load(f)

centroids = results["centroids"]
paths = results["paths"]

st.title("Traffic-Based Route Guidance System")
source = results["source"]
target = results["target"]
timestamp = results["timestamp"]
model = results["model"]
routes = results["routes"]



if paths:
    # Center map on source
    m = folium.Map(location=centroids[source], zoom_start=13)
    colors = ["red", "blue", "green", "purple", "orange"]
    for node_id, (lat, lon) in centroids.items():
        folium.Marker(
            location=(lat, lon),
            popup=f"Node {node_id}<br>({lat:.5f}, {lon:.5f})",
            icon=folium.Icon(color="gray", icon="info-sign")
        ).add_to(m)


    for idx, path_info in enumerate(paths):
        path_nodes = path_info[0]
        points = [centroids[n] for n in path_nodes if n in centroids]
        folium.PolyLine(points, color=colors[idx % len(colors)], weight=5, opacity=0.7).add_to(m)
        for n in path_nodes:
            folium.CircleMarker(
                location=centroids[n],
                radius=5,
                color=colors[idx % len(colors)],
                fill=True,
                fill_color=colors[idx % len(colors)],
                popup=n
            ).add_to(m)
    st_folium(m, width=700, height=500)

    # Show route details
    for idx, path_info in enumerate(paths):
        st.markdown(f"### 🛣️ Route {idx+1}")
        st.write(" → ".join(path_info[0]))
        st.write(f"📏 Total distance: {path_info[2]:.2f} km")
        st.write(f"⏱️ Total travel time: {path_info[1]:.1f} minutes")
else:
    st.error("No route found.")

