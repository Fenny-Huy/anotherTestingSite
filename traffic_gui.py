import sys
import types

# Patch torch.classes to avoid Streamlit watcher error
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is not None:
    import torch
    if not hasattr(torch, "classes"):
        torch.classes = types.SimpleNamespace()

import streamlit as st
import folium
from streamlit_folium import st_folium
import datetime
from algorithms.graph_builder import build_graph
from route_finder import find_routes

# Load centroids for dropdowns
centroids, _ = build_graph('data/scats_complete_average.csv')

st.title("Traffic-Based Route Guidance System")

# Sidebar inputs
source = st.selectbox("Source Node", options=centroids.keys())
target = st.selectbox("Target Node", options=centroids.keys())
day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=25)
time = st.time_input("Time", value=datetime.time(8, 30))
model = st.selectbox("Model", options=["LSTM", "GRU", "MLP", "TCN"])
routes = st.number_input("Number of Routes", min_value=1, max_value=5, value=3)

if st.button("Find Routes"):
    # Compose timestamp
    timestamp = f"2006-10-{day:02d} {time.strftime('%H:%M:%S')}"
    centroids, paths = find_routes(
        source, target, timestamp, model, routes,
        nodes_path='data/scats_complete_average.csv',
        volumes_path='data/traffic_model_ready.pkl'
    )

    if paths:
        # Center map on source
        m = folium.Map(location=centroids[source], zoom_start=13)
        colors = ["red", "blue", "green", "purple", "orange"]
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