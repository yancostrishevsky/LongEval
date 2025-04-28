import os
import json

def preprocess_documents(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for doc_file in os.listdir(input_dir):
        if doc_file.endswith('.jsonl'):
            with open(os.path.join(input_dir, doc_file), 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    indexable = {
                        'id': doc['id'],
                        'contents': doc['title'] + ' ' + doc.get('abstract', '')
                    }
                    with open(os.path.join(output_dir, f"{doc['id']}.json"), 'w') as outf:
                        json.dump(indexable, outf)

if __name__ == "__main__":
    preprocess_documents('path/to/documents/', 'path/to/index_input/')
