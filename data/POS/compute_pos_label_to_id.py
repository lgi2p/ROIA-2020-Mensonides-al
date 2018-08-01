import io

pos_labels_to_id = dict()

with io.open('pos_label_embeddings.txt') as f:
    lines = f.read().splitlines()

for line in lines:
    if len(line) != 0:
        pos_label = line.split(' ')[0]
        if pos_label not in pos_labels_to_id:
            pos_labels_to_id[pos_label] = len(pos_labels_to_id)

with open('pos_labels_to_id.txt', 'w') as f:
    for key in pos_labels_to_id:
        tbw = str(key)+'\t'+str(pos_labels_to_id.get(key))+'\n'
        f.write(tbw)