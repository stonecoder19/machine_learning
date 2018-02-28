
def entropy(class_probabilities):
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)


def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilites = class_probabilities(labels)
    return entropy(probabilities)

def partitiion_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / 
                               total_count for subset in subsets)



def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    
    return groups

def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


for key in ['level','lang','tweets','phd']:
    print key, partition_entropy_by(inputs, key)


senior_inputs = [(input, label) for input, label in inputs if input["level"] == "Senior"]

for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy(senior_inputs, key)


def classify(tree, input):
    if tree in [True, False]:
        return tree


    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)
    

    if subtree_key not in subtree_dict:
        subtree_key = None

    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)


def build_tree_id3(inputs, split_canidates=None):
    if split_candidates is None:
        split_canidates = input[0][0].keys()

    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues


    if num_trues == 0: return False
    if num_falses == 0: return True

    if not split_canidates:
        return num_trues >= num_falses


    best_attribute = min(split_canidates, 
                         key=partial(partition_entropy_by,inputs))

    partitions = partition_by(inputs, best_attribute)
    new_canidates = [a for a in split_candidates
                    if a!= best_attribute]

    subtrees = { attribute_values: build_tree_id3(subset,new_candidates)
                 for attribute_value, subset in partition.iteritems()}
    
    subtrees[None] = num_trees > num_falses

    return (best_attribute, subtrees)
