from utils import permutations, euclidean_distance

cities_distances = {'Toronto': (10, 10), 'Scotia': (9, 8), 'Quebec': (15, 12), 'Montreal': (20, 20)}
cities = cities_distances.keys()
distances = cities_distances.values()

all_cities = permutations(list(cities))

selected_idx = 0
selected_distance = None
for idx, city in enumerate(all_cities):
    d_array = [cities_distances[c] for c in city]
    d_cities = [euclidean_distance(d, d_array[i+1])for i, d in enumerate(d_array) if i+1 < len(d_array)]
    distance = sum(d_cities)
    if idx != 0 and distance < selected_distance:
        selected_idx = idx
        selected_distance = distance
    if idx == 0:
        selected_distance = distance

selected_path = all_cities[selected_idx]
print(f"The optimal path is {selected_path} and total distance for this path is {selected_distance}")
