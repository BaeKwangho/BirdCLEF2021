import argparse
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preparing for Birdcall dataset"
    )
    
    parser.add_argument(
        "-d",
        "--dirs",
        type=str,
        help="Directory where data stored. specise folders have to be located under this directory path.",
        default='./asset/birdclef-2021/train_short_audio'
    )
    
    parser.add_argument(
        "-n",
        "--num_data",
        type=int,
        help="Define how many frames in one bridcall file",
        default=500
    )
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    epochs = args.epochs

if __name__ == "__main__":
    main()
