from run_model import run_model
import subprocess
import argparse

def main():
    p = argparse.ArgumentParser(
        description="TBRGS: Traffic‑Based Route Guidance System"
    )
    p.add_argument('--source',    required=True, help='Origin site ID (e.g. 0970)')
    p.add_argument('--target',    required=True, help='Destination site ID (e.g. 3685)')
    p.add_argument('--timestamp', required=True,
                   help='Timestamp for prediction (YYYY-MM-DD HH:MM:SS)')
    p.add_argument('--model', required=True, help = 'Model to use (LSTM or GRU)')
    p.add_argument('--routes', help = 'How many routes to return')
    p.add_argument('--map', default=False)
    args = p.parse_args()

    paths = run_model(args.source, args.target, args.timestamp, args.model, int(args.routes))

    if bool(args.map):
        subprocess.run(["streamlit", "run", "map_only.py"])
        
if __name__ == "__main__":
    main()