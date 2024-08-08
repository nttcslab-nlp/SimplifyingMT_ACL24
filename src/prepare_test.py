import datasets as ds
import pandas as pd
from tap import Tap

from utils import *

class Args(Tap):
    dataset_name: str = "cl-nagoya/Simplifyingmt" 

    target_age: int = 10
    output_dir_path: str = ""

def main(args):
    dataset = ds.load_dataset(args.dataset_name,split='test')
    df = pd.DataFrame(dataset)

    AoA = create_aoa_dict()

    ref = df['reference'].to_list()
    hyp = df['hypothesis'].to_list()
    src = df['source'].to_list()

    hyp_aoas = get_aoa(hyp,AoA)[1]
    ref_aoas = get_aoa(ref,AoA)[1]

    for i in range(len(ref)):
        if max(hyp_aoas[i]) >= args.target_age:
            if max(ref_aoas[i]) >= args.target_age:
                df.loc[i,'target'] = "delete_line"
        else:
            df.loc[i,'target'] = "delete_line"

    df = df[~df['target'].apply(lambda x: "delete_line" in x)]

    out_path = args.output_dir_path + "/test1.jsonl"
    df.to_json(out_path,orient='records',force_ascii=False,lines = True)
    out_path_ref = args.output_dir_path + "/reference.txt"
    with open(out_path_ref, "w") as f:
        f.write("\n".join(ref))
    out_path_src = args.output_dir_path + "/source.txt"
    with open(out_path_src, "w") as f:
        f.write("\n".join(src))


if __name__=="__main__":
    args = Args().parse_args()
    main(args)