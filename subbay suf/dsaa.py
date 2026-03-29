# First DSA Program in Python
# Topic: Array (List)

# Take size of array
n = int(input("Enter number of elements: "))

# Create empty array
arr = []

# Take array elements from user
for i in range(n):
    element = int(input(f"Enter element {i+1}: "))
    arr.append(element)

# Display the array
print("\nArray elements are:")
for i in range(n):
    print(arr[i], end=" ")

# Find sum of elements
total = 0
for i in range(n):
    total += arr[i]

print("\nSum of array elements:", total)