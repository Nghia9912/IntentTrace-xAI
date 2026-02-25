# main.py
import argparse
from benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="IntentTrace-xAI: LLM Code Evaluation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Execution mode")

    subparsers.add_parser("benchmark", help="Run AUC-ROC evaluation on Ground Truth dataset")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()