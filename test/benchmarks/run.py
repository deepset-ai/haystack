from retriever import benchmark_indexing, benchmark_querying
from reader import benchmark_reader
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

args = parser.parse_args()

if args.retriever_index:
    # test DB connection
    from haystack.document_store.faiss import FAISSDocumentStore
    document_store = FAISSDocumentStore(sql_url="postgresql://postgres:password@localhost:5432/haystack")
    document_store.get_all_documents()
    benchmark_indexing(args.ci)
if args.retriever_query:
    benchmark_querying(args.ci)
if args.reader:
    benchmark_reader(args.ci)

