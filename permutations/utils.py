import numpy as np
def permutations(items):
    if len(items) <= 1:
        return [items]

    all_permutations = []
    for idx in range(len(items)):
        current_item = items[idx]
        remaining_items = items[:idx] + items[idx+1:]

        for perm in permutations(remaining_items):
            all_permutations.append([current_item] + perm)
    return all_permutations

def euclidean_distance(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

if __name__ == "__main__":
    mylist = [1, 2, 3]
    nlist = permutations(mylist)
    print(nlist)

    l1 = [1, 2, 3]
    l2 = [4, 5, 2]
    print(euclidean_distance(l1, l2))