import pandas as pd
from tap import Tap

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils 

class Args(Tap):
    test_dir_path:str=""
    generation_dir_path:str =""
    output_dir_path:str = ""

def main(args):

    def make_df(df_file, text_file):
        df = pd.read_json(df_file, lines = True)
        generation = utils.read_lines_from_file(text_file)
        df['generation'] = generation

        return df
    
    def update_and_save(base_df, new_df):
        updated_df = base_df.merge(new_df, on='reference', how='left', suffixes=('', '_new'))
        updated_df['generation'] = updated_df['generation_new'].combine_first(updated_df['generation'])
        updated_df = updated_df.drop(columns=['generation_new'])
        current_df = updated_df

        return current_df

    def save_ref_src(df,out_path):
        src,ref = df['source'].to_list(),df['reference'].to_list()
        
     
    def save_generation(df,out_path):
        generations = df['generation'].to_list()
        with open(out_path, "w") as f:
            f.write("\n".join(generations))
    
    for name in range(1,6):
        df_path = args.test_dir_path + str(name) + ".jsonl"
        gene_path = args.generation_dir_path + str(name) + ".txt"
        out_path = args.output_dir_path + str(name) + ".txt"
        print(df_path,gene_path)
        df = make_df(df_path, gene_path)

        if name == 1:
            current_df = df
            save_ref_src(current_df, out_path)
            save_generation(current_df, out_path)
        else:
            current_df = update_and_save(current_df, df)
            save_generation(current_df, out_path)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
