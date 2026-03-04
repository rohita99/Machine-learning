# Training dataset
data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
]

# Separate attributes and target
concepts = [row[:-1] for row in data]
target = [row[-1] for row in data]

num_attr = len(concepts[0])

# Initialize Specific and General hypothesis
S = ['0'] * num_attr
G = [['?'] * num_attr]

print("Initial S:", S)
print("Initial G:", G)

for i in range(len(concepts)):
    
    if target[i] == "Yes":   # Positive Example
        
        for j in range(num_attr):
            if S[j] == '0':
                S[j] = concepts[i][j]
            elif S[j] != concepts[i][j]:
                S[j] = '?'
        
        G = [g for g in G if all(g[k] == '?' or g[k] == S[k] for k in range(num_attr))]
    
    else:   # Negative Example
        
        new_G = []
        for g in G:
            for j in range(num_attr):
                if g[j] == '?':
                    if concepts[i][j] != S[j]:
                        temp = g.copy()
                        temp[j] = S[j]
                        new_G.append(temp)
        G = new_G

print("\nFinal Specific Hypothesis (S):", S)

print("\nFinal General Hypothesis (G):")
for g in G:
    print(g)
