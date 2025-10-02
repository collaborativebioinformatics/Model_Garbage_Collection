from src.gnn.lcilp.utils.data_utils import process_files

# locate graph files associated with their name 
files = {
    'train': './lcilp/data/alzheimers_triples.txt'
}

if __name__ == '__main__':
    graph = process_files(files)