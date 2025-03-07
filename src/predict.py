import sys
sys.path.insert(0, sys.path[0]+"/../")

from api import Predictor
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='cuda', 
                      help='device, cuda/cpu, default is cuda')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size. Default=1')
    parser.add_argument('--species', '-s', type=str, default='4mC_C.equisetifolia',
                      help='Default is 4mC_C.equisetifolia')
    parser.add_argument('--input_file', '-i', type=str, required=True,
                      help='input file path')

    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = [line.strip() for line in f if line.strip()]

    predictor = Predictor(device=args.device, batch_size=args.batch_size)
    output = predictor.predict(args.species, data)
    
    print("Prediction result: ")
    for seq, result in zip(data, output):
        print(f"Sequence: {seq[:15]}... => Result (is methylated or not): {'Yes' if result == 1 else 'No'}")

if __name__ == "__main__":
    main()