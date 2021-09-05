# The benchmarks use
# - a variant of the Natural Questions Dataset (https://ai.google.com/research/NaturalQuestions) from Google Research
#   licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
# - the SQuAD 2.0 Dataset (https://rajpurkar.github.io/SQuAD-explorer/) from  Rajpurkar et al.
#   licensed under  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/legalcode)

from retriever import benchmark_indexing, benchmark_querying
from reader import benchmark_reader
from utils import load_config
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--reader', default=False, action="store_true",
                    help='Perform Reader benchmarks')
parser.add_argument('--retriever_index', default=False, action="store_true",
                    help='Perform Retriever indexing benchmarks')
parser.add_argument('--retriever_query', default=False, action="store_true",
                    help='Perform Retriever querying benchmarks')
parser.add_argument('--ci', default=False, action="store_true",
                    help='Perform a smaller subset of benchmarks that are quicker to run')
parser.add_argument('--update_json', default=False, action="store_true",
                    help='Update the json file with the results of this run so that the website can be updated')
parser.add_argument('--save_markdown', default=False, action="store_true",
                    help='Update the json file with the results of this run so that the website can be updated')
args = parser.parse_args()

# load config
params, filenames = load_config(config_filename="config.json", ci=args.ci)

if args.retriever_index:
    benchmark_indexing(**params, **filenames, ci=args.ci, update_json=args.update_json, save_markdown=args.save_markdown)
if args.retriever_query:
    benchmark_querying(**params, **filenames, ci=args.ci, update_json=args.update_json, save_markdown=args.save_markdown)
if args.reader:
    benchmark_reader(**params, **filenames, ci=args.ci, update_json=args.update_json, save_markdown=args.save_markdown)

