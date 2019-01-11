import numpy as np, matplotlib.pyplot as plt
import random, operator
from itertools import groupby

def encode(input_string):
    return [(len(list(g))) for k,g in groupby(input_string)]
 
def decode(lst):
    b = ''.join('01'*(len(lst)/2))
    r = ''
    for i in range(len(lst)):
        r += lst[i] * b[i]
    return r

#Creates a binary matrix with size 'size' and probability of a 1 occuring as 'probability'
def create_matrix(size, probability): 
    return np.random.choice(a=[1, 0], size=(size, size), p=[probability, 1 - probability])

def fitness(matrix, rule_list, supergeneration):
    
    if supergeneration%2:
        matrix3 = margolus1(matrix, rule_list)
        return count_alive1(matrix3)
    else:
        matrix3 = margolus2(matrix, rule_list)
        return count_alive2(matrix3)
    #return len(encode(''.join([str(i) for n in matrix3 for i in n])))

def margolus1(matrix, rule_list):
    matrix_size = len(matrix)
    matrix2 = matrix.copy()
    #print(matrix2)

    for x in range(0,matrix_size,2):
        for y in range(0,matrix_size,2):

            #Create a pointer for each 2x2 cell in the matrix. Nested array eg: [[0, 1], [1, 1]]
            square = matrix2[x:x+2,y:y+2]

            #Find the current configuration of the cell. an int ranging from 0 to 15. eg: 14
            config = int(''.join([str(j) for i in square for j in i][::-1]), 2)

            #Find config of the next gen of cell according to given RCA rule. an array  eg: [0, 1, 0, 1]
            values = [int(k) for k in str(bin(rule_list[config]))[:1:-1].ljust(4, '0')]

            #Set the new config for the cell eg: [[0, 1], [0, 1]] 
            square[0,0], square[0,1] = values[0], values[1]
            square[1,0], square[1,1] = values[2], values[3]

    return matrix2
    

def margolus2(matrix, rule_list):
    matrix2 = margolus_shift1(matrix)
    matrix2 = margolus1(matrix2, rule_list)
    return margolus_shift2(matrix2)

def margolus_shift1(matrix):
    shiftedup =  np.roll(matrix, -1, axis=0)
    shiftedleft = np.roll(shiftedup, -1, axis=1)
    return shiftedleft

def margolus_shift2(matrix):
    shiftedup =  np.roll(matrix, 1, axis=0)
    shiftedleft = np.roll(shiftedup, 1, axis=1)
    return shiftedleft

def count_alive1(matrix):
    """Counts number of cells that will be alive in the next generation of cellular automata"""
    matrix3 = matrix.copy()
    matrix3 = margolus_shift1(matrix3)
    matrix_size = len(matrix3)
    
    counter = 0
    for x in range(0,matrix_size,2):
        for y in range(0,matrix_size,2): #Starts the 2X2 box at 0,0 and then moves onto the next 2X2 box
            square = matrix3[x:x+2,y:y+2] #Creates a pointer to the 2X2 part of the dis
            #print("square = ", square)
            if 1 in square:
                counter += 1

    return counter

def count_alive2(matrix):
    """Counts number of cells that will be alive in the next generation of cellular automata"""
    matrix3 = matrix.copy()
    #matrix3 = margolus_shift1(matrix3)
    matrix_size = len(matrix3)
    
    counter = 0
    for x in range(0,matrix_size,2):
        for y in range(0,matrix_size,2): #Starts the 2X2 box at 0,0 and then moves onto the next 2X2 box
            square = matrix3[x:x+2,y:y+2] #Creates a pointer to the 2X2 part of the dis
            #print("square = ", square)
            if 1 in square:
                counter += 1

    return counter

def initialise_population(population_size):
    a=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    population = []
    
    for i in range(population_size):
        population.append(random.sample(a, len(a)))
    
    return population

def rank_population(matrix, population, supergeneration):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = fitness(matrix.copy(), population[i], supergeneration)
    
    sorted_fitness = sorted(fitnessResults.items(), key = operator.itemgetter(1))
    best = sorted_fitness[0][0]
    #print "Best Chromosome: ", population[best], sorted_fitness[0]
    #print "Best Chromosome: ", sorted_fitness[0]
    
    ranked_population = []

    for i in sorted_fitness:
        index = i[0]
        ranked_population.append(population[index])
            
    return ranked_population, sorted_fitness[0][1], population[best]

def breed(parent1, parent2):
    """Return a new chromosome created via partially matched crossover (PMX).
    This is suitable for permutation encoded GAs. Partially matched crossover
    respects the absolute position of alleles.
    Args:
        parent1 (List): A parent chromosome.
        parent2 (List): A parent chromosome.
    Returns:
        List[List]: Two new chromosomes descended from the given parents.
    Source:
        https://github.com/rawg/levis/blob/master/levis/crossover.py#L271
    """
    third = len(parent1) // 3
    l1 = int(random.triangular(1, third, third * 2))
    l2 = int(random.triangular(third, third * 2, len(parent1) - 1))

    if l2 < l1:
        l1, l2 = l2, l1

    def pmx(parent1, parent2):
        matching = parent2[l1:l2]
        displaced = parent1[l1:l2]
        child = parent1[0:l1] + matching + parent1[l2:]

        tofind = [item for item in displaced if item not in matching]
        tomatch = [item for item in matching if item not in displaced]

        for k, v in enumerate(tofind):
            subj = tomatch[k]
            locus = parent1.index(subj)
            child[locus] = v

        return child

    #print "Parents ", parent1, "and ", parent2, "breeded ", pmx(parent1, parent2)
    return [pmx(parent1, parent2), pmx(parent2, parent1)]

def mating_pool(ranked_population, elite_size):
    
    elite_size = int(len(ranked_population) * elite_size)
    elite = ranked_population[0:elite_size]
    
    breeders_size = int(len(ranked_population)/(2 *elite_size))
    breeders = ranked_population[0:breeders_size]
    
    #print elite_size
    #print breeders_size
    
    children = []
    
    for parent1 in elite:
        for parent2 in breeders:
            offsprings = breed(parent1, parent2)
            children.extend(offsprings)
            
    #children.extend(initialise_population(500))
    return children

def mutate(ruleset, probability):
    """Return a mutated chromosome made through swap mutation.
    This chooses two random points in a list and swaps their values. It is
    suitable for use with list, value, and permutation encoding, but designed
    for the latter.
    Args:
        chromosome (List): The source chromosome for mutation.
        probability (float): The mutation rate or the probability that an
            allele will be mutated.
    Returns:
        List: A mutant chromosome.
    Source:
        https://github.com/rawg/levis/blob/master/levis/mutation.py#L75
    """
    
    probability /= 14 #Added this line later given difference between given and actual mutation rate
    mutant = list(ruleset)
    changes = 0
    offset = probability

    for locus1 in range(0, len(ruleset)):
        if random.random() < offset:
            locus2 = locus1
            while locus2 == locus1:
                locus2 = random.randint(0, len(ruleset) - 1)

            mutant[locus1], mutant[locus2] = mutant[locus2], mutant[locus1]
            changes += 2
            offset = probability / changes

    return mutant

def mutate_population(population, mutation_rate):
    mutated_population = []
    
    #print "size of population to mutate: ", len(population)
    for ruleset in range(0, len(population)):
        mutated_ruleset = mutate(population[ruleset], mutation_rate)
        mutated_population.append(mutated_ruleset)
    return mutated_population

def next_generation(matrix, population, elite_size, mutation_rate, supergeneration):
    ranked_population, best_fitness, best_chromosome = rank_population(matrix, population, supergeneration)
    children = mating_pool(ranked_population, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen, best_fitness, best_chromosome

def genetic_algorithm(matrix_size, population_size, elite_size, mutation_rate, generations, supergeneration):
    matrix = create_matrix(matrix_size, 0.5)
    population = initialise_population(population_size)

    for i in range(0, generations):
        population, best_fitness, best_chromosome = next_generation(matrix, population, elite_size, mutation_rate, supergeneration)

    return best_fitness

from math import factorial
from itertools import count

def to_factoradic(n):
    """Convert an integer to a factoradic number.
    Args:
        n: A non-negative integer to be converted.
    Returns:
        A sequence of integers where the factorial of each zero-based
        index gives a place value, and the item at that index is the
        coefficient by which the place value is to be multiplied. The
        sum of the multiples of the factorial place values is equal to
        the argument, n.  Since the coefficient at any index must be
        less that or equal to the index, the coefficient at index 0
        is always 0.
    Raises:
        ValueError: If n is negative.
        ValueError: If n is not integral.
        ValueError: If n is not finite.
    """
    if n < 0:
        raise ValueError("Negative number {} cannot be represented "
                         "as a factoradic number".format(n))

    try:
        v = int(n)
    except OverflowError:
        raise ValueError("Non-finite number {} cannot be represented "
                         "as a factoradic number".format(n))
    else:
        if v != n:
            raise ValueError("Non-integral {} cannot be represented "
                             "as a factoradic number".format(n))

    quotient = n
    coefficients = []
    for radix in count(start=1):
        quotient, remainder = divmod(quotient, radix)
        coefficients.append(remainder)
        if quotient == 0:
            break
    #return coefficients[::-1]
    return padzeroes(coefficients[::-1])

def padzeroes(coefficients):
    while len(coefficients) != 16:
        coefficients.insert(0, 0)
    return coefficients

def from_factoradic(coefficients):
    """Convert a sequence of factoradic coefficients to an integer.
    Args:
        coefficients: A sequence of integers where the factorial of
            each zero-based index gives a place value, and the item
            at that index is the coefficient by which the place value
            is to be multiplied.
    Returns:
        The integer equivalent of the factoradic representation.
    Raises:
        ValueError: If coefficients does not contain at least
            one element.
        ValueError: If not all elements in coefficients are
            less than or equal to their index values.
    """
    if len(coefficients) < 1:
        raise ValueError("coefficients {!r} does not contain at least one element".format(coefficients))
    if any(coefficient > index for index, coefficient in enumerate(coefficients)):
        raise ValueError("Not all coefficients in {!r} are less than or "
                         "equal to their index values.".format(coefficients))
    return sum(factorial(i)*v for i, v in enumerate(coefficients))

def nth(n):
    origin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    new = []
    og = origin[:]
    f = to_factoradic(n)
    #print(f)
    for i in f:
        #print(i, og)
        new.append(og[i])
        del og[i]
        #print(new, og)
        
    og.extend(new)
    return og

def permutationth(rule_list):
    origin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    c = 1
    n = 0
    for i in rule_list:
        #print(n, i, len(rule_list) - c)
        n += origin.index(i) * factorial(len(rule_list) - c)
        origin.remove(i)
        c += 1
    
    return n

def genetic_algorithm_m(matrix, population_size, elite_size, mutation_rate, generations, supergeneration):
    population = initialise_population(population_size)
    progress = []
    
    for i in range(0, generations):
        population, best_fitness, best_chromosome = next_generation(matrix, population, elite_size, mutation_rate, supergeneration)
        progress.append(best_fitness)
        
    print(permutationth(best_chromosome))
    
    if supergeneration%2:
        matrix = margolus1(matrix, best_chromosome)
    else:
        matrix = margolus2(matrix, best_chromosome)

    return matrix, best_fitness, progress

def iterative_genetic_algorithm(supergenerations):

	matrix_size     = 10
	population_size = 500
	elite_size      = 0.02
	mutation_rate   = 0.05

	matrix = create_matrix(matrix_size, 0.5)
	print(matrix)

	print("Original length: ", len(''.join([str(i) for n in matrix for i in n])))
	print("Original RLE length: ", len(encode(''.join([str(i) for n in matrix for i in n]))))
	progress = []
	for supergeneration in range(0, supergenerations):
		matrix, best_fitness, midprogress = genetic_algorithm_m(matrix, population_size, elite_size, mutation_rate, 7, supergeneration)
		#print(matrix)

		progress.extend(midprogress)

	print(progress)

	print("Final RLE length: ", len(encode(''.join([str(i) for n in matrix for i in n]))))
	print(matrix)

	return matrix, best_fitness, progress

print(iterative_genetic_algorithm(10))