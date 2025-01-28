import csv
import time
import random
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from distill_utils import distill
from surrogate import predictor
from flops import TransformerHparams
from reduction import reduction_z3

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

def convert_chromosomes(population):
    """
        Converts the chromosomes of a given population into a specific format suitable for surrogate modeling.

        This function iterates through each individual in the population, rounding certain genes to a specified
        precision and ensuring others are integers. Specifically, the genes at indices 5 and 8 are rounded to
        one decimal place, while the rest are rounded to the nearest integer.

        Parameters:
        - population (list of lists): A population where each individual is represented as a list of genes (chromosomes).
                                      Each gene in an individual can be a float or an integer.

        Process:
        1. Iterates through each individual in the population.
        2. For each gene in an individual:
           - If the gene's index is 5 or 8, it is rounded to one decimal place and added to a temporary list.
           - Otherwise, the gene is rounded to the nearest integer and added to the temporary list.
        3. The modified individual is then added to a new list representing the converted population.

        Returns:
        - surrogate_data (list of lists): A new population list where each individual's genes have been converted
                                          according to the specified rules. This format is typically used for
                                          preparing data for surrogate models in optimization tasks.

        Note:
        - The function assumes the population list is well-formed, with each individual containing at least 9 genes.
        - Indices are 0-based, meaning the 6th and 9th genes are at indices 5 and 8, respectively.
    """
    surrogate_data = []
    for each_pop in population:
        typed_chromosome = []
        for idx in range(len(each_pop)):
            if idx == 5 or idx == 8:
                typed_chromosome.append(round(each_pop[5], 1))
            else:
                typed_chromosome.append(int(round(each_pop[idx])))

        surrogate_data.append(typed_chromosome)
    return surrogate_data

def hyperparams_convert_back(hyperparams):
    # Reverse mappings
    tokenizer_type_inv = {"BPE": 1, "WordPiece": 2, "Unigram": 3, "Word": 4}
    hidden_act_inv = {"gelu": 1, "relu": 2, "silu": 3, "gelu_new": 4}
    position_embedding_type_inv = {"absolute": 1, "relative_key": 2, "relative_key_query": 3}
    learning_rate_inv = {1e-3: 1, 1e-4: 2, 5e-5: 3}
    batch_size_inv = {16: 1, 32: 2, 64: 3, 8: 4}

    return [
        tokenizer_type_inv[hyperparams[0]],  # Reverse conversion for 'Tokenizer'
        hyperparams[1],  # 'Vocab Size' remains the same (assuming it's numeric)
        hyperparams[2],  # 'Num Hidden Layers' remains the same (assuming it's numeric)
        hyperparams[3],  # 'Hidden Size' remains the same (assuming it's numeric)
        hidden_act_inv[hyperparams[4]],  # Reverse conversion for 'Hidden Act'
        hyperparams[5],  # 'Hidden Dropout Prob' remains the same (assuming it's numeric)
        hyperparams[6],  # 'Intermediate Size' remains the same (assuming it's numeric)
        hyperparams[7],  # 'Num Attention Heads' remains the same (assuming it's numeric)
        hyperparams[8],  # 'Attention Probs Dropout Prob' remains the same (assuming it's numeric)
        hyperparams[9],  # 'Max Sequence Length' remains the same (assuming it's numeric)
        position_embedding_type_inv[hyperparams[10]],  # Reverse conversion for 'Position Embedding Type'
        learning_rate_inv[hyperparams[11]],  # Reverse conversion for 'Learning Rate'
        batch_size_inv[hyperparams[12]]  # Reverse conversion for 'Batch Size'
    ]

class Candidate(object):
    def __init__(self, candidates_vals):
        self.candidate_values = candidates_vals
        self.objective_values = []

    def get_candidate_values(self):
        return self.candidate_values

    def set_objective_values(self, objective_values):
        self.objective_values = objective_values

    def get_objective_values(self):
        return self.objective_values

    def set_candidate_values_at_index(self, indx, val):
        self.candidate_values[indx] = val

def generate_random_population(size, lb, ub):
    random_pop = []

    for i in range(size):
        while True:
            candidate_vals = []
            for index in range(len(lb)):
                if index == 5 or index == 8:
                    candidate_vals.append(round(random.uniform(lb[index], ub[index]+0.1), 1))
                else:
                    candidate_vals.append(int(random.uniform(lb[index], ub[index]+1)))
            if candidate_vals[3] % candidate_vals[7] == 0:
                break
        random_pop.append(Candidate(candidate_vals))

    return random_pop

def calculate_minimum_distance(candidate, random_pop):
    distance = 1e9

    for each_candidate in random_pop:
        vals = each_candidate.get_candidate_values()
        candidate_vals = candidate.get_candidate_values()
        dist = np.linalg.norm(np.array(vals) - np.array(candidate_vals))
        if dist < distance:
            distance = dist

    return distance

def generate_adaptive_random_population(size, lb, ub):
    random_pop = []

    initial_pop = generate_random_population(10, lb, ub)[0]

    random_pop.append(initial_pop)

    while len(random_pop) < size:
        D = 0
        selected_candidate = None
        rp = generate_random_population(size, lb, ub)
        for each_candidate in rp:
            min_dis = calculate_minimum_distance(each_candidate, random_pop)
            if min_dis > D:
                D = min_dis
                selected_candidate = each_candidate

        random_pop.append(selected_candidate)

    return random_pop

def evaulate_population(population, surrogate_model):
    for each_candidate in population:
        candidate_values = each_candidate.get_candidate_values()
        model = TransformerHparams(candidate_values[3], candidate_values[2], candidate_values[9], candidate_values[1], candidate_values[6], candidate_values[7])
        size = abs(model.get_params()*4/1e6 - 3)
        flops = model.get_infer_flops()/1e9
        accuracy = surrogate_model.predict([candidate_values])[0]
        fitnesses = [size, accuracy, flops]
        each_candidate.set_objective_values(fitnesses)

def dominates(candidate1, candidate2):
    candidate1_objectives = candidate1.get_objective_values()
    candidate2_objectives = candidate2.get_objective_values()
    dominates = False
    if candidate1_objectives[0] < candidate2_objectives[0] and candidate1_objectives[1] > candidate2_objectives[1] and candidate1_objectives[2] < candidate2_objectives[2]:
        dominates = True
    return dominates

def update_archive(pop, archive):
    for each_candidate in pop:
        dominated = False
        for each_archive in archive:
            if dominates(each_archive, each_candidate):
                dominated = True
                break
        if not dominated:
            if len(archive) == 0:
                archive.append(each_candidate)
            else:
                to_remove = []
                for each_archive in archive:
                    if dominates(each_candidate, each_archive) or each_archive.get_candidate_values() == each_candidate.get_candidate_values():
                        to_remove.append(each_archive)

                for each_remove in to_remove:
                    archive.remove(each_remove)
                archive.append(each_candidate)

def partially_mapped_crossover(parent1, parent2):
    parent1_values = parent1.get_candidate_values()
    parent2_values = parent2.get_candidate_values()
    size = len(parent1_values)
    child1 = [-1] * size
    child2 = [-1] * size

    crossover_point1, crossover_point2 = sorted(random.sample(range(size), 2))

    for i in range(crossover_point1, crossover_point2 + 1):
        if parent2_values[i] not in child1:
            next_index = i
            while child1[next_index] != -1:
                next_index = parent2_values.index(parent1_values[next_index])
            child1[next_index] = parent2_values[i]

        if parent1_values[i] not in child2:
            next_index = i
            while child2[next_index] != -1:
                next_index = parent1_values.index(parent2_values[next_index])
            child2[next_index] = parent1_values[i]

    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2_values[i]
        if child2[i] == -1:
            child2[i] = parent1_values[i]

    return Candidate(child1), Candidate(child2)

def boundary_random_mutation(candidate, lb, ub, thresh):
    candidate_values = candidate.get_candidate_values()
    for index in range(len(candidate_values)):
        if random.uniform(0, 1) < thresh:
            if index == 5 or index == 8:
                candidate_values[index] = round(random.uniform(lb[index], ub[index]+0.1), 1)
            else:
                candidate_values[index] = int(random.uniform(lb[index], ub[index]+1))

    return candidate

def correct(pop, lb, ub):
    for indx in range(len(pop)):
        candidate = pop[indx]
        values = candidate.get_candidate_values()
        # for value_index in range(len(values)):
        #     pop[indx].set_candidate_values_at_index(value_index, int(pop[indx].get_candidate_values()[value_index]))
        #     while values[value_index] > ub[value_index] or values[value_index] < lb[value_index]:
        #         temp = generate_random_population(1, lb, ub)[0]
        #         pop[indx].set_candidate_values_at_index(value_index, int(temp.get_candidate_values()[value_index]))

        while values[3] % values[7] != 0:
            temp = generate_random_population(1, lb, ub)[0]
            pop[indx].set_candidate_values_at_index(3, int(temp.get_candidate_values()[3]))

    return pop

def select_best(tournament_candidates):
    best = tournament_candidates[0]
    for i in range(len(tournament_candidates)):
        candidate1 = tournament_candidates[i]
        for j in range(len(tournament_candidates)):
            candidate2 = tournament_candidates[j]
            if (dominates(candidate1, candidate2)):
                best = candidate1
    return best

def tournament_selection(pop, size):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates)
    return best

def generate_off_springs(pop, lb, ub):
    size = len(pop)
    population_to_return = []

    while len(population_to_return) < size:
        parent1 = tournament_selection(pop, 10)
        parent2 = tournament_selection(pop, 10)
        while parent1 == parent2:
            parent2 = tournament_selection(pop, 10)

        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= 0.60:
            parent1, parent2 = partially_mapped_crossover(parent1, parent2)
        child1 = boundary_random_mutation(parent1, lb, ub, 0.1)
        child2 = boundary_random_mutation(parent2, lb, ub, 0.1)

        population_to_return.append(child1)
        population_to_return.append(child2)

    return population_to_return

if __name__ == "__main__":
    start_time = time.time()
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    # lb, ub = reduction_z3()
    # logging.info("Time taken for reduction: {}".format(time.time() - start_time))
    # logging.info("Lower Bound: {}".format(lb))
    # logging.info("Upper Bound: {}".format(ub))
    lb = [1, 1000, 1, 16, 1, 0.2, 32, 1, 0.2, 256, 1, 1, 1]
    ub = [4, 46000, 12, 256, 4, 0.5, 3072, 12, 0.5, 512, 3, 3, 2]



    #surrogate_data = []
    #for each_pop in pop:
    #    surrogate_data.append(each_pop.get_candidate_values())

    #accs = distill(surrogate_data, surrogate=True)

    #with open("surrogate_train_data_20.csv", "w") as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob", "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length", "Position Embedding Type", "Learning Rate", "Batch Size", "Accuracy"])
    #    for d, acc in zip(surrogate_data, accs):
    #        writer.writerow(hyperparams_convert(d) + [acc])

    # Read the CSV file into a DataFrame
    df = pd.read_csv("surrogate_data_metamorphic.csv")

    # Apply the conversion function to each row, excluding the last column
    df.iloc[:, :-2] = df.iloc[:, :-2].apply(lambda row: hyperparams_convert_back(row.tolist()), axis=1,
                                            result_type='expand')
    surrogate_data = df.iloc[:, :-2].values
    accs = df['Accuracy'].tolist()
    features2 = df.iloc[:, -1].values
    flips = df['Prediction Flips'].tolist()

    surrogate_model = predictor([surrogate_data, accs])

    pop = generate_adaptive_random_population(50, lb, ub)
    evaulate_population(pop, surrogate_model)
    archive = []
    update_archive(pop, archive)

    iteration = 100
    for i in tqdm(range(iteration)):
        pop = generate_off_springs(pop, lb, ub)
        pop = correct(pop, lb, ub)
        evaulate_population(pop, surrogate_model)
        update_archive(pop, archive)

    logging.info("Time taken: {}".format(time.time() - start_time))
    logging.info("Number of solutions in the archive: {}".format(len(archive)))
    logging.info("Saving the archive to the file")
    #with open("pareto_set.csv", "w") as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob", "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length", "Position Embedding Type", "Learning Rate", "Batch Size", "Size", "Accuracy", "FLOPS"])
    #    for each_archive in archive:
    #        writer.writerow(each_archive.get_candidate_values() + each_archive.get_objective_values())

    # Extract the first objective
    closest_index = -1
    closest_size = +1000000
    for i in range(len(archive)):
        objs = archive[i].get_objective_values()
        if objs[0] < closest_size:
            closest_index = i
            closest_size = objs[0]
    # Retrieve the solution with the first objective closest to 3

    closest_solution = archive[closest_index].get_candidate_values()
    print("solution = ", closest_solution)
    objs = archive[closest_index].get_objective_values()
    logging.info("Objs : {}".format(objs))
    converted_sol = convert_chromosomes([closest_solution])
    accs, prediction_flips = distill([closest_solution], eval=False, surrogate=False)
    accs, prediction_flips = distill([closest_solution], eval=True, surrogate=False)
    logging.info("Prediction flips : {}".format(prediction_flips))

    # Field names for the CSV
    fieldnames = [
        "Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob",
        "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length",
        "Position Embedding Type", "Learning Rate", "Batch Size", "Size", "Accuracy", "FLOPS", "Flips"
    ]

    results_file = "so_avatar_results.csv"
    with open(results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Create a dictionary from row data
        row_data = {
            "Tokenizer": converted_sol[0][0],
            "Vocab Size": converted_sol[0][1],
            "Num Hidden Layers": converted_sol[0][2],
            "Hidden Size": converted_sol[0][3],
            "Hidden Act": converted_sol[0][4],
            "Hidden Dropout Prob": converted_sol[0][5],
            "Intermediate Size": converted_sol[0][6],
            "Num Attention Heads": converted_sol[0][7],
            "Attention Probs Dropout Prob": converted_sol[0][8],
            "Max Sequence Length": converted_sol[0][9],
            "Position Embedding Type": converted_sol[0][10],
            "Learning Rate": converted_sol[0][11],
            "Batch Size": converted_sol[0][12],
            "Size": (3+objs[0]),  # Assuming objs[0] is the Size
            "Accuracy": accs[0],  # Assuming accs contains accuracy values
            "FLOPS": objs[2],  # Assuming objs[2] is the FLOPS
            "Flips": prediction_flips[0]  # Assuming prediction_flips contains the flips value
        }

        # Write the row data to the CSV file
        writer.writerow(row_data)