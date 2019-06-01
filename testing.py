from DataComp import *

def main():
    print("Testing Compression Code")


    #Encode and Decode Functions
    print("Testing Run Length Encoding Function with input: 000011101010010101110010101110001111111")
    print(encode("000011101010010101110010101110001111111"))
    print("Output should be [4, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3]")
    
    print("Testing Run Length Decoding Function with input [4, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3]")
    print(decode([4, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 3, 3, 7]))
    print("Output should be '000011101010010101110010101110001111111'")

    
    #Matrix Create Function
    print("Testing Matrix Create Function with 4x4 size and half probability of 1's")
    matrix = create_matrix(4, 0.5)
    print(matrix)
    print("Matrix size = ", len(matrix))
    print("Count of 1's = ", matrix.sum())

    #Create Population Function
    print("Testing Function to create random set of population of 10 RCA rules")
    pop = initialise_population(100)
    print(pop)
    
    #Ranking Function
    print("Testing Function to rank each RCA Rule in created population based on fitness function")
    ranked, best, bestrule = rank_population(matrix, pop, 3)
    print(ranked)
    print("best, best rule = ", best, bestrule)

    #Mating Pool Function
    print("Testing Function to create a mating pool for all RCA rules with elite percentage of 1%")
    print(mating_pool(ranked, 0.01))

    #Mutate Function
    print("Testing mutate function to make sure it mutates according to given mutation rate")
    counter = 0
    mutation_rate = 0.02
    for i in range(1000):
        rule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        if mutate(rule, mutation_rate) != rule:
            counter += 1
    print("Given Mutation Rate: ", mutation_rate)
    print("Actual Mutation Rate: ", counter/1000.0)

    #Factoradic Functions
    print("Testing Factoradic Function with input 677644")
    print(nth(677644))
    print("Output should be [0, 1, 2, 3, 4, 5, 7, 14, 13, 10, 8, 6, 9, 15, 11, 12]")
    print("Testing Reverse Factoradic Function with input [0, 1, 2, 3, 4, 5, 7, 14, 13, 10, 8, 6, 9, 15, 11, 12]")
    print(permutationth([0, 1, 2, 3, 4, 5, 7, 14, 13, 10, 8, 6, 9, 15, 11, 12]))
    print("Output should be 677644")

    #Genetic Algorithm
    print("Testing genetic algorithm with a 20x20 matrix, 1000 population, 1% elite size, and 5% mutation rate for 6 generations")
    print(genetic_algorithm_plot(20, 1000, 0.01, 0.05 , 6, 1))
    
    #Iterative Genetic Algorithm
    print("Testing iterative genetic algorithm for a 20x20 matix for 8 iterations with a 1000 population")
    iterative_genetic_algorithm_plot(8, 20, 1000)
    
if __name__ == "__main__":
    main()
