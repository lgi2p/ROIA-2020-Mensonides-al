import io

labels_to_id = dict()

def update_label_dict(label_dict, file_to_read):
    with io.open(file_to_read) as f:
        lines = f.read().splitlines()

    for line in lines:
        if len(line)>0 and line[0].isdigit():
            infos = line.split('\t')
            label = infos[7].split(':')[0]
            if label not in label_dict:
                label_dict[label] = len(label_dict)

update_label_dict(labels_to_id, 'UD_English-EWT/en_ewt-ud-train.conllu')
update_label_dict(labels_to_id, 'UD_English-EWT/en_ewt-ud-test.conllu')
update_label_dict(labels_to_id, 'UD_English-EWT/en_ewt-ud-dev.conllu')

update_label_dict(labels_to_id, 'UD_English-GUM/en_gum-ud-train.conllu')
update_label_dict(labels_to_id, 'UD_English-GUM/en_gum-ud-test.conllu')
update_label_dict(labels_to_id, 'UD_English-GUM/en_gum-ud-dev.conllu')

update_label_dict(labels_to_id, 'UD_English-LinES/en_lines-ud-train.conllu')
update_label_dict(labels_to_id, 'UD_English-LinES/en_lines-ud-test.conllu')
update_label_dict(labels_to_id, 'UD_English-LinES/en_lines-ud-dev.conllu')

update_label_dict(labels_to_id, 'UD_English-PUD/en_pud-ud-test.conllu')

update_label_dict(labels_to_id, 'UD_English-ParTUT/en_partut-ud-train.conllu')
update_label_dict(labels_to_id, 'UD_English-ParTUT/en_partut-ud-test.conllu')
update_label_dict(labels_to_id, 'UD_English-ParTUT/en_partut-ud-dev.conllu')

with open('dep_parsing_labels_to_id.txt', 'w') as f:
    for key in labels_to_id:
            tbw = str(key)+'\t'+str(labels_to_id.get(key))+'\n'
            f.write(tbw)












# with io.open('pos_label_embeddings.txt') as f:
#     lines = f.read().splitlines()
#
# for line in lines:
#     if len(line) != 0:
#         pos_label = line.split(' ')[0]
#         if pos_label not in labels_to_id:
#             labels_to_id[pos_label] = len(labels_to_id)
#
# with open('pos_labels_to_id.txt', 'w') as f:
#     for key in labels_to_id:
#         tbw = str(key)+'\t'+str(labels_to_id.get(key))+'\n'
#         f.write(tbw)