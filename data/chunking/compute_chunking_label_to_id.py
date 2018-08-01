import io

chunking_labels_to_id = dict()

with io.open('chunking_label_embeddings.txt') as f:
    lines = f.read().splitlines()

for line in lines:
    if len(line) != 0:
        chunking_label = line.split(' ')[0]
        if chunking_label not in chunking_labels_to_id:
            chunking_labels_to_id[chunking_label] = len(chunking_labels_to_id)

with open('chunking_labels_to_id.txt', 'w') as f:
    for key in chunking_labels_to_id:
        tbw = str(key)+'\t'+str(chunking_labels_to_id.get(key))+'\n'
        f.write(tbw)