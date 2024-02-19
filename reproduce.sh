#!/bin/bash

# Run all Oracles and create their data
python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=true
python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=lucky
python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=unlucky
python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=best-3
python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=best-4
python3 run_oracle.py --path=data/TSB/metrics/ --acc=all --randomness=best-5
