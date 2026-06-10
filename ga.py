import EasyGA, optimize
from params import FREE, BOUNDS
ga = EasyGA.GA()
ga.gene_impl   = lambda: None  # use chromosome_impl below instead
ga.chromosome_length = len(FREE)
ga.chromosome_impl = lambda: [__import__('random').uniform(*BOUNDS[k]) for k in FREE]
ga.fitness_function_impl = lambda chrom: optimize.fitness([g.value for g in chrom.gene_list])
ga.evolve()