import sys
sys.argv = [
    "main_run.py",
    "--folder=bgl/", "--log_file=BGL.log", "--dataset_name=bgl", "--model_name=logrobust",
    "--window_type=sliding", "--sample=sliding_window", "--is_logkey", "--semantics", "--input_size=300",
    "--train_size=0.8", "--train_ratio=1", "--valid_ratio=0.1", "--test_ratio=1",
    "--max_epoch=100", "--n_warm_up_epoch=0", "--n_epochs_stop=10", "--batch_size=64",
    "--num_candidates=9", "--history_size=10", "--lr=0.001", "--accumulation_step=1",
    "--session_level=entry", "--window_size=20", "--step_size=20",
    "--output_dir=experimental_results/full/random/", "--device=mps",
]

import os
from main_run import arg_parser
from logadempirical.logdeep.tools.predict import Predicter

parser = arg_parser()
args = parser.parse_args()

args.data_dir = os.path.expanduser(args.data_dir + args.folder)
args.output_dir += args.folder

options = vars(args)
if options['session_level'] == "entry":
    options["output_dir"] = options["output_dir"] + str(int(options["window_size"])) + "/"

options["model_dir"] = options["output_dir"] + options["model_name"] + "/"
options["train_vocab"] = options["output_dir"] + "train.pkl"
options["vocab_path"] = options["output_dir"] + options["model_name"] + "_vocab.pkl"
options["model_path"] = options["model_dir"] + options["model_name"] + ".pth"
options["scale_path"] = options["model_dir"] + "scale.pkl"

Predicter(options).predict_supervised2()
