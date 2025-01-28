import csv

from mo_distill_utils import distill, hyperparams_convert
from flops import TransformerHparams
from many_objective import convert_chromosomes, MyRepair, ModelCompressionProblem, \
    LatinHypercubeSampler


if __name__ == "__main__":
    # Define the lower and upper bounds
    lb = [1, 1000, 1, 16, 1, 0.2, 32, 1, 0.2, 256, 1, 1, 1]
    ub = [4, 46000, 12, 256, 4, 0.5, 3072, 12, 0.5, 512, 3, 3, 2]

    # Number of points to generate
    n_points = 20

    problem = ModelCompressionProblem(lb, ub, None)
    sampler = LatinHypercubeSampler()

    surrogate_data = convert_chromosomes(sampler._do(problem, 20))

    # trains the models
    accs, prediction_flips = distill(surrogate_data, eval=False, surrogate=True)

    print("Create surrogate models")

    with open("surrogate_data_metamorphic.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob",
             "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length",
             "Position Embedding Type", "Learning Rate", "Batch Size", "Model Size","Accuracy", "Prediction Flips"])
        for i in range(0, len(accs)):
            model = TransformerHparams(surrogate_data[i][3], surrogate_data[i][2], surrogate_data[i][9],
                                       surrogate_data[i][1], surrogate_data[i][6], surrogate_data[i][7])
            size = abs(model.get_params() * 4 / 1e6)

            row_data = hyperparams_convert(surrogate_data[i])
            row_data += [size]
            row_data += [accs[i]]
            row_data += [prediction_flips[i]]
            writer.writerow(row_data)