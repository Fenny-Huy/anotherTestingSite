from algorithms.graph_builder import build_graph
from algorithms.astar_search import astar
from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
from utils.edge_mapper import EdgeMapper

def find_routes(source, target, timestamp, model, routes, nodes_path, volumes_path):
    centroids, edges = build_graph(nodes_path)
    if model.upper() == 'LSTM':
        predictor = LSTMPredictor(data_pkl=volumes_path, models_dir="lstm_saved_models")
    elif model.upper() == 'GRU':
        predictor = GRUPredictor(data_pkl=volumes_path, models_dir="gru_saved_models")
    elif model.upper() == 'MLP':
        predictor = MLPPredictor(data_pkl=volumes_path, models_dir="mlp_saved_models")
    elif model.upper() == 'TCN':
        predictor = TCNPredictor(data_pkl=volumes_path, models_dir="tcn_saved_models")
    else:
        raise ValueError("Unknown model")
    paths = astar(source, target, centroids, edges, predictor, timestamp, k=routes)
    return centroids, paths