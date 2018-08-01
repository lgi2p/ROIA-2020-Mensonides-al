import io

# create a file that will be used to train a pos_label embedding

examples = []
new_example = []


with io.open('train.txt') as f:
    lines = f.read().splitlines()

for line in lines:
    if len(line) == 0:
        if len(new_example) != 0:
            examples.append(new_example)
        new_example = []

    else:
        pos_label = line.split(' ')[1]
        new_example.append(pos_label)


new_example = []
with io.open('test.txt') as f:
    lines = f.read().splitlines()

for line in lines:
    if len(line) == 0:
        if len(new_example) != 0:
            examples.append(new_example)
        new_example = []

    else:
        pos_label = line.split(' ')[1]
        new_example.append(pos_label)




with open('only_pos_labels.txt', 'w') as f:
    for example in examples:
        for pos_label in example:
            f.write(pos_label)
            f.write(' ')
        f.write('\n')