import pandas as pd
from readability import Readability
from tap import Tap
from evaluate import load

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils 

class Args(Tap):
    generate_file:str = ""
    test_file:str = ""

def main(args:Args):

    generations = utils.read_lines_from_file(args.generate_file)
    sources,references = utils.df_to_src_ref(args.test_file)
    references = [ [ex] for ex in references]

    text = " ".join(generations)
    r = Readability(text)

    sari = load("sari")
    sari_score = sari.compute(sources=sources,predictions=generations,references=references)

    fk = r.flesch_kincaid()
    fk_score = fk.score

    dc = r.dale_chall()
    dc_score = dc.score

    print("sari_score, fkgl_score, dc_score")
    print(next(iter(sari_score.values())),fk_score,dc_score)

if __name__ == '__main__' :
    args = Args().parse_args()
    main(args)