from tap import Tap

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils 

class Args(Tap):
    generate_file:str = ""

def main(args:Args):
    AoA = utils.create_aoa_dict()
    
    max_aoa_values = utils.get_aoa(utils.read_lines_from_file(args.generate_file),AoA)[1]
    max_aoa_counts = utils.count_aoa(max_aoa_values)
    average_max_aoa = utils.average_aoa(max_aoa_values)

    num_success_sentence = sum(max_aoa_counts[:10])
    
    print("Success Rate")
    print(num_success_sentence/len(max_aoa_values))
    print('Average AoA')
    print(average_max_aoa)

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)