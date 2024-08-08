from tap import Tap
import pandas as pd

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

class Args(Tap):
    input_test_file:str = ""
    generation_file:str = ""
    output_test_file:str = ""

def main(args: Args):
    AoA = utils.create_aoa_dict()

    generations = utils.read_lines_from_file(args.generation_file)
    generate_words ,generate_aoas = utils.get_aoa(generations,AoA)
    sources,references = utils.df_to_src_ref(args.input_test_file)
    targets=[]
    
    for i in range(len(generations)):
        tmp =[]
        idx = generate_aoas[i].index(max(generate_aoas[i]))
        if max(generate_aoas[i]) >= 10:
            tmp.append(generate_words[i][idx])
        else:
            tmp.append("delete_line")
        targets.append(tmp)

    df = pd.DataFrame({'source':sources,'target':targets,'hypothesis':generations,'reference':references})
    df = df[~df['target'].apply(lambda x: "delete_line" in x)]
    df.to_json(args.output_test_file,orient='records',force_ascii=False,lines = True)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)