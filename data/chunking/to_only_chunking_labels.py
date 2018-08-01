import io

# create a file that will be used to train a chunking_label embedding

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
        chunking_label = line.split(' ')[2]
        new_example.append(chunking_label)


new_example = []
with io.open('test.txt') as f:
    lines = f.read().splitlines()

for line in lines:
    if len(line) == 0:
        if len(new_example) != 0:
            examples.append(new_example)
        new_example = []

    else:
        chunking_label = line.split(' ')[2]
        new_example.append(chunking_label)




with open('only_chunking_labels.txt', 'w') as f:
    for example in examples:
        for chunking_label in example:
            f.write(chunking_label)
            f.write(' ')
        f.write('\n')