import time
import numpy as np
from math import sin, cos, acos, radians

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Reproducibility
np.random.seed(5) # <- comment out for random seed
#print("Seed: ", np.random.get_state()[1][0]) 


cityCoords = {'Paris':(48.856667, 2.350833),
              'Marseille':(43.296386, 5.369954),
              'Lyon':(45.759723, 4.842223),
              'Toulouse':(43.604503, 1.444026),
              'Nice':(43.703393, 7.266274),
              'Strasbourg':(48.584445, 7.748612),
              'Nantes':(47.21806, -1.55278),
              'Bordeaux':(44.838611, -0.578333),
              'Montpellier':(43.61194, 3.87722),
              'Rennes':(48.114722, -1.679444),
              'Lille':(50.637222, 3.063333),
              'Le Havre':(49.498889, 0.121111),
              'Reims':(49.26278, 4.03472),
              'Saint-Étienne':(45.434722, 4.390278),
              'Toulon':(43.125, 5.930556),
              'Limoges':(45.8302431, 1.2584025),
              'Brest':(48.3881950, -4.4889750),
              'Rouen':(49.4392914, 1.0901092),
              'Amiens':(49.8940647, 2.2987044),
              'Saint-Malo':(48.6494894, -2.0255400),
              'La Rochelle':(46.1590039, -1.1495397),
              'Avignon':(43.9491728, 4.8068550),
              'Dijon':(47.3217692, 5.0426494),
              'Troyes':(48.2953894, 4.0729100),
              'Orleans':(47.9008144, 1.9027247),
              'Tours':(47.3946114, 0.6836900),
              'Caen':(49.1863789, -0.3629728),
              'Calais':(50.9587725, 1.8530175),
              'Metz':(49.1202156, 6.1765236),
              'Nancy':(48.6891036, 6.1810664),
              'Grenoble':(45.1908439, 5.7294236),
              'Clermont-Ferrand':(45.7764986, 3.0815433),
              'Aix-en-Provence':(43.5280, 5.4471),
              'Béziers':(43.34302, 3.21524),
              'Perpignan':(42.69874, 2.89573),
              'Pau':(43.2961, -0.3701),
              'Cahors':(44.4476, 1.4418),
              'Agen':(44.20483, 0.62360),
              'Valence':(44.9327, 4.8950),
              'Chambéry':(45.5651, 5.9236),
              'Aix-les-Bains':(45.68966, 5.91593),
              'Annecy':(45.8976, 6.1264),
              'Sallanches':(45.93445, 6.63176),
              'Passy':(45.91812, 6.70517),
              'Megève':(45.85665, 6.61881),
              'Albertville':(45.67120, 6.38728),
              'Mâcon':(46.30399, 4.83428),
              'Chalon-sur-Saône':(46.7820, 4.8595),
              'Besançon':(47.2358, 6.0287),
              'Belfort':(47.6398, 6.8577),
              'Beauvais':(49.43073, 2.08718),
              'Cambrai':(50.1759, 3.2383),
              'Arras':(50.29090, 2.78047),
              'Creil':(49.26089, 2.47733),
              'Abbeville':(50.1059, 1.8401),
              'Vernon':(49.0925, 1.4859),
              'Dreux':(48.73587, 1.36710),
              'Chartres':(48.44515, 1.48808),
              'Angers':(47.4713, -0.5524),
              'Cholet':(47.06084, -0.87979),
              'Poitiers':(46.58283, 0.34416),
              'Dinard':(48.63314, -2.05711),
              'Quimper':(47.99639, -4.10343),
              'Challans':(46.84750, -1.87701),
              'Chauvigny':(46.56890, 0.64769),
              'Montluçon':(46.34035, 2.60645),
              'Nevers':(46.9958, 3.1635),
              'Bourges':(47.0837, 2.3968),
              'Vierzon':(47.22341, 2.06850),
              'Moulins':(46.56571, 3.33316),
              'Cusset':(46.12449, 3.42483),
              'Roanne':(46.03663, 4.07373),
              'Aurillac':(44.9293, 2.4507),
              'Rodez':(44.3482, 2.5771),
              'Carcassone':(43.21348, 2.35276),
              'Castres':(43.60522, 2.24255),
              'Tarbes':(43.23279, 0.08044),
              'Anglet':(43.48137, -1.51279),
              'Cambo-les-Bains':(43.36017, -1.40164),
              'Lourdes':(43.0964, -0.0445),
              'Albi':(43.92773, 2.14732),
              'Bergerac':(44.85248, 0.48450),
              'Périgueux':(45.18860, 0.71852),
              'Ruffec':(46.02800, 0.19989),
              'Bellac':(46.12156, 1.05066),
              'Bressuire':(46.84170, -0.49018),
              'Vire':(48.83851, -0.88749),
              'Flers':(48.74843, -0.56867),
              'Morlaix':(48.5813, -3.8305),
              'Lannion':(48.7317, -3.4584),
              'Lorient':(47.7475, -3.3672),
              'Vannes':(47.6570, -2.7570),
              'Les-Sables-d\'Olonne':(46.49779, -1.78219),
              'La Roche-sur-Yon':(46.66977, -1.42562),
              'Niort':(46.32430, -0.46181),
              'La Souterraine':(46.2361, 1.4883),
              'Auxerre':(47.7956, 3.5720),
              'Tonnerre':(47.85544, 3.97383),
              'Toul':(48.67526, 5.89104),
              'Antibes':(43.58247, 7.12237)
}

def calcCityDistances(coordDict):

    cities = list(coordDict.keys())
    n = len(cities)
    distances = {}

    for i in range(n - 1):
        cityA = cities[i]
        latA, longA = coordDict[cityA]
        latA = radians(latA)
        longA = radians(longA)

        for j in range(i + 1, n):
            cityB = cities[j]
            latB, longB = coordDict[cityB]
            latB = radians(latB)
            longB = radians(longB)
            dLong = abs(longA - longB)
            angle = acos(sin(latA)*sin(latB)+cos(latA)*cos(latB)*cos(dLong))
            dist = angle * 6371.1  #   Mean    Earth   radius  (km)
            key = frozenset((cityA, cityB))
            distances[key] = dist

    return  distances

def getRouteLength(distanceData, route):
    
    distance = 0.0
    for i, pointA in enumerate(route[:-1]):
        pointB = route[i+1]
        key = frozenset((pointA,  pointB))
        distance += distanceData[key]

    return distance

last_was_improvement = False
def print_improvement(dist, it):
    global last_was_improvement
    print(f"\rImproved in iteration [{it}/{generations}]: distance = {dist:8.2f}km", end="".ljust(20))
    last_was_improvement = True

def print_stagnation(generation):
    global last_was_improvement
    if last_was_improvement: print()
    print(f"-> Applying 2-opt at generation {generation} due to stagnation.".ljust(80)) 
    last_was_improvement = False
    
def plot_route_on_map(cityCoords, route):
    """Plot the TSP route on a map of France using Cartopy."""
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add borders and coastline for context
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Set the extent (longitude and latitude limits) for France
    ax.set_extent([-5, 10, 41, 51])

    # Plot each city as a point and route as a line
    for city, coords in cityCoords.items():
        ax.plot(coords[1], coords[0], marker='o', color='red' if city in route else 'blue', markersize=5, transform=ccrs.Geodetic())
        ax.text(coords[1] + 0.1, coords[0], city, fontsize=4, transform=ccrs.Geodetic())

    # Connect the route
    route_lons = [cityCoords[city][1] for city in route] + [cityCoords[route[0]][1]]
    route_lats = [cityCoords[city][0] for city in route] + [cityCoords[route[0]][0]]
    ax.plot(route_lons, route_lats, color='darkorange', linewidth=2, marker='o', markersize=3, transform=ccrs.Geodetic())

    route_dist = getRouteLength(calcCityDistances(cityCoords), route)
    
    plt.title(f"TSP Route over France (Total Distance: {route_dist:.2f} km)")
    plt.show()


def mutate_route(route_a, route_b, route_c):
    """
    Create a mutant route by selecting a segment from route_b and inserting it into route_a,
    inspired by the DE mutation strategy but adapted for discrete TSP.
    """
    size = len(route_a)
    # Select two random cut points for the segment to be inserted
    start, end = np.sort(np.random.choice(range(size), 2, replace=False))
    # Extract the segment from route_b
    segment = route_b[start:end]
    # Remove cities in the segment from route_a
    route_a_reduced = [city for city in route_a if city not in segment]
    # Insert segment into a random position in route_a_reduced
    insert_pos = np.random.randint(0, len(route_a_reduced))
    mutant_route = route_a_reduced[:insert_pos] + segment + route_a_reduced[insert_pos:]
    return mutant_route

def recombine(route_original, route_mutant):
    """Recombine the original and mutant route using a simple crossover strategy."""
    size = len(route_original)
    # Select a crossover point
    cross_point = np.random.randint(0, size - 1)
    # Initialize the new route with None
    trial_route = [None] * size
    # Copy a segment from route_original to the trial_route
    trial_route[:cross_point] = route_original[:cross_point]
    # Fill the rest with cities from route_mutant in the order they appear (avoiding duplicates)
    fill_pos = cross_point
    for city in route_mutant:
        if city not in trial_route:
            trial_route[fill_pos] = city
            fill_pos += 1
            if fill_pos == size:
                break
    return trial_route

def two_opt(route, distanceData):
    """ 
    2-opt algorithm for improving an existing route.
    The 2-opt algorithm works by iteratively removing two edges and reconnecting the two paths in all possible ways.
    If the reconnection results in a shorter route, the change is accepted.
    """ 
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # Neighboring edges, no need to check
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1]  # Reverse the segment
                if getRouteLength(distanceData, new_route) < getRouteLength(distanceData, route):
                    route = new_route
                    improved = True
        if not improved:
            break
    return route

def handle_stagnation(population, fitness, distanceData, generation):
    """Applies 2-opt optimization across the population to overcome stagnation at a specified generation."""
    print_stagnation(generation)
    for i in range(len(population)):
        population[i] = two_opt(population[i], distanceData)
        fitness[i] = getRouteLength(distanceData, population[i])
        
    return population, fitness

def solve_TSP(distanceData, cities, generations, popsize):
    
    # Initialize population with random routes
    population = [np.random.permutation(cities).tolist() for _ in range(popsize)]
    
    # Calculate fitness for each route
    fitness = [getRouteLength(distanceData, route) for route in population]
    
    # Find the best route in the initial population
    best_index = np.argmin(fitness)
    best_fitness = fitness[best_index]
    
    # Initialize stagnation counter and threshold
    stagnation_counter = 0
    stagnation_threshold = 110
    
    for generation in range(generations):
        for i in range(popsize):
            # Select three random routes from the population
            indices = [j for j in range(popsize) if j != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            # Create a mutant route by combining routes a, b, and c
            mutant_route = mutate_route(population[a], population[b], population[c])
            # Recombine the mutant route with the original route
            trial_route = recombine(population[i], mutant_route)
            trial_fitness = getRouteLength(distanceData, trial_route)
            
            if trial_fitness < fitness[i]:
                population[i], fitness[i] = trial_route, trial_fitness
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    if best_fitness < 6000: # Early stopping 
                        print(f"Found a route with distance < 6000 km at generation!")
                        print_improvement(best_fitness, generation)
                        return population[i], fitness[i]
                    print_improvement(fitness[i], generation)
                    stagnation_counter = 0 # Reset counter on improvement

        # Apply 2-opt if stagnation is detected
        if stagnation_counter >= stagnation_threshold:
            population, fitness = handle_stagnation(population, fitness, distanceData, generation)
            stagnation_counter = 0 # Reset counter after applying 2-opt
        else: 
            stagnation_counter += 1

    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]


if __name__ == "__main__":
    # Parameters 
    generations = 10000
    popsize = 100
    distances = calcCityDistances(cityCoords)
    cities = list(cityCoords.keys())
    
    print("Solving TSP...")
    print("-----------------------------------------------------------")
    start_time = time.time() 
    result = solve_TSP(distances, cities, generations, popsize)
    end_time = time.time()

    route = result[0]
    distance = result[1]
    print()
    print("-----------------------------------------------------------")
    print(f"\nBest route found:\n{route}\nDistance: {distance:.2f} km")
    print(f"Process took {(end_time - start_time)/60:.2f} minutes.")

    plot_route_on_map(cityCoords, route)

## NOTE: Best routes found for 10000 generations and popsize 100
# route_5991 = ['Sallanches', 'Passy', 'Megève', 'Albertville', 'Annecy', 'Aix-les-Bains', 'Chambéry', 'Grenoble', 'Valence', 'Saint-Étienne', 'Lyon', 'Roanne', 'Mâcon', 'Chalon-sur-Saône', 'Dijon', 'Besançon', 'Belfort', 'Strasbourg', 'Nancy', 'Toul', 'Metz', 'Reims', 'Cambrai', 'Arras', 'Lille', 'Calais', 'Abbeville', 'Amiens', 'Beauvais', 'Creil', 'Paris', 'Troyes', 'Tonnerre', 'Auxerre', 'Nevers', 'Moulins', 'Cusset', 'Clermont-Ferrand', 'Montluçon', 'Bourges', 'Vierzon', 'Orleans', 'Chartres', 'Dreux', 'Vernon', 'Rouen', 'Le Havre', 'Caen', 'Flers', 'Vire', 'Rennes', 'Saint-Malo', 'Dinard', 'Lannion', 'Morlaix', 'Brest', 'Quimper', 'Lorient', 'Vannes', 'Nantes', 'Challans', "Les-Sables-d'Olonne", 'La Roche-sur-Yon', 'La Rochelle', 'Niort', 'Bressuire', 'Cholet', 'Angers', 'Tours', 'Chauvigny', 'Poitiers', 'Ruffec', 'Bellac', 'La Souterraine', 'Limoges', 'Périgueux', 'Bergerac', 'Bordeaux', 'Anglet', 'Cambo-les-Bains', 'Pau', 'Lourdes', 'Tarbes', 'Agen', 'Cahors', 'Aurillac', 'Rodez', 'Albi', 'Toulouse', 'Castres', 'Carcassone', 'Perpignan', 'Béziers', 'Montpellier', 'Avignon', 'Aix-en-Provence', 'Marseille', 'Toulon', 'Antibes', 'Nice']
# route_5977 = ['Passy', 'Sallanches', 'Megève', 'Albertville', 'Annecy', 'Aix-les-Bains', 'Chambéry', 'Grenoble', 'Valence', 'Saint-Étienne', 'Lyon', 'Roanne', 'Mâcon', 'Chalon-sur-Saône', 'Dijon', 'Besançon', 'Belfort', 'Strasbourg', 'Nancy', 'Toul', 'Metz', 'Reims', 'Troyes', 'Tonnerre', 'Auxerre', 'Nevers', 'Moulins', 'Cusset', 'Clermont-Ferrand', 'Montluçon', 'Bourges', 'Vierzon', 'Orleans', 'Chartres', 'Dreux', 'Vernon', 'Paris', 'Creil', 'Beauvais', 'Amiens', 'Arras', 'Cambrai', 'Lille', 'Calais', 'Abbeville', 'Rouen', 'Le Havre', 'Caen', 'Flers', 'Vire', 'Rennes', 'Saint-Malo', 'Dinard', 'Lannion', 'Morlaix', 'Brest', 'Quimper', 'Lorient', 'Vannes', 'Nantes', 'Challans', "Les-Sables-d'Olonne", 'La Roche-sur-Yon', 'La Rochelle', 'Niort', 'Bressuire', 'Cholet', 'Angers', 'Tours', 'Chauvigny', 'Poitiers', 'Ruffec', 'Bellac', 'La Souterraine', 'Limoges', 'Périgueux', 'Bergerac', 'Bordeaux', 'Anglet', 'Cambo-les-Bains', 'Pau', 'Lourdes', 'Tarbes', 'Agen', 'Cahors', 'Aurillac', 'Rodez', 'Albi', 'Toulouse', 'Castres', 'Carcassone', 'Perpignan', 'Béziers', 'Montpellier', 'Avignon', 'Aix-en-Provence', 'Marseille', 'Toulon', 'Antibes', 'Nice']