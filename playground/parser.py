import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_number', type=str, 
                    help="run number corresponding to model used for analyses")
parser.add_argument('--folder', type=str, 
                    help="Folder where model is saved")

args = parser.parse_args()
print(args.run_number)
print(args.folder)
