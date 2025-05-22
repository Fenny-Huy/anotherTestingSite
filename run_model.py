#!/usr/bin/env python3
# main.py

import argparse
from algorithms.graph_builder import build_graph
from algorithms.astar_search   import astar
from utils.edge_mapper         import EdgeMapper
from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
from utils.flow_to_speed       import flow_to_speed

import sys
import json


def run_model(source, target, timestamp, model, routes):


    # 1) Build graph
    print(f"🔍 Building graph from 'data/scats_complete_average.csv' …")
    centroids, edges = build_graph('data/scats_complete_average.csv')
    print(f"{centroids}")

    # 2) Instantiate mapper & predictor
    print("🗺️  Initializing edge→arm mapper & LSTM predictor …")
    mapper    = EdgeMapper('data/traffic_model_ready.pkl')
    if model.upper() == 'LSTM':
        predictor = LSTMPredictor(data_pkl='data/traffic_model_ready.pkl',
                                  models_dir="lstm_saved_models")
    elif model.upper() == 'GRU':
        predictor = GRUPredictor(data_pkl='data/traffic_model_ready.pkl',
                                 models_dir="gru_saved_models")
    elif model.upper() == 'MLP':
        predictor = MLPPredictor(data_pkl='data/traffic_model_ready.pkl',
                                 models_dir="mlp_saved_models")
    elif model.upper() == 'TCN':
        predictor = TCNPredictor(data_pkl='data/traffic_model_ready.pkl',
                                 models_dir="tcn_saved_models")

    # 4) Run A* to get the fastest route under predicted traffic
    print(f"🚦 Running A* from {source} → {target} at {timestamp} …")
    #path, total_time = astar(args.source, args.target,centroids, edges, predictor, args.timestamp)
    paths = astar(source, target, centroids, edges, predictor, timestamp, k=routes)



    results = {
    "source": source,
    "target": target,
    "timestamp": timestamp,
    "model": model.upper(),
    "routes": routes,
    "paths": paths,  # list of ([nodes], time, distance)
    "centroids": centroids
    }
    with open("results.json", "w") as f:
        json.dump(results, f)

    return paths

