import math

# Dataset
data = [
    ['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes'],
    ['Rain','Cool','Normal','Strong','No'],
    ['Overcast','Cool','Normal','Strong','Yes'],
    ['Sunny','Mild','High','Weak','No'],
    ['Sunny','Cool','Normal','Weak','Yes'],
    ['Rain','Mild','Normal','Weak','Yes'],
    ['Sunny','Mild','Normal','Strong','Yes'],
    ['Overcast','Mild','High','Strong','Yes'],
    ['Overcast','Hot','Normal','Weak','Yes'],
    ['Rain','Mild','High','Strong','No']
]

labels = ['Outlook','Temperature','Humidity','Wind']

# Function to calculate entropy
def entropy(data):
    total = len(data)
    yes = sum(1 for row in data if row[-1] == 'Yes')
    no = total - yes
    
    if yes == 0 or no == 0:
        return 0
    
    p_yes = yes/total
    p_no = no/total
    
    return -p_yes*math.log2(p_yes) - p_no*math.log2(p_no)

# Function to calculate information gain
def info_gain(data, attr):
    total_entropy = entropy(data)
    values = set(row[attr] for row in data)
    
    weighted_entropy = 0
    
    for v in values:
        subset = [row for row in data if row[attr] == v]
        weighted_entropy += (len(subset)/len(data))*entropy(subset)
    
    return total_entropy - weighted_entropy

# ID3 algorithm
def id3(data, labels):
    
    classes = [row[-1] for row in data]
    
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    
    if len(labels) == 0:
        return max(set(classes), key=classes.count)
    
    gains = [info_gain(data, i) for i in range(len(labels))]
    best_attr = gains.index(max(gains))
    
    tree = {labels[best_attr]:{}}
    
    values = set(row[best_attr] for row in data)
    
    for v in values:
        subset = [row for row in data if row[best_attr] == v]
        new_labels = labels[:]
        new_labels.pop(best_attr)
        
        reduced_subset = [row[:best_attr]+row[best_attr+1:] for row in subset]
        
        subtree = id3(reduced_subset, new_labels)
        tree[labels[best_attr]][v] = subtree
    
    return tree

# Build tree
tree = id3(data, labels)

print("Decision Tree:")
print(tree)

# Classify new sample
def classify(tree, labels, sample):
    
    if type(tree) != dict:
        return tree
    
    attr = list(tree.keys())[0]
    attr_index = labels.index(attr)
    
    value = sample[attr_index]
    
    subtree = tree[attr][value]
    
    new_sample = sample[:attr_index] + sample[attr_index+1:]
    new_labels = labels[:]
    new_labels.pop(attr_index)
    
    return classify(subtree, new_labels, new_sample)

# New sample
sample = ['Sunny','Cool','High','Strong']

result = classify(tree, labels, sample)

print("\nNew Sample:", sample)
print("Classification:", result)
