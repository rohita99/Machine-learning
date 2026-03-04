# Training dataset
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Number of attributes
num_attributes = len(data[0]) - 1

# Initialize hypothesis
hypothesis = ['0'] * num_attributes

print("Initial Hypothesis:", hypothesis)

# FIND-S Algorithm
for instance in data:
    if instance[-1] == "Yes":
        for i in range(num_attributes):
            if hypothesis[i] == '0':
                hypothesis[i] = instance[i]
            elif hypothesis[i] != instance[i]:
                hypothesis[i] = '?'

print("\nFinal Hypothesis:", hypothesis)
