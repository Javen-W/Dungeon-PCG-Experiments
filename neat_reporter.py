import logging
import time
from neat.reporting import BaseReporter
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys


class ResultsReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, logger: logging.Logger):
        self.show_species_detail = True
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.logger = logger

    def start_generation(self, generation):
        self.generation = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            self.logger.info('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            self.logger.info("   ID   age  size  fitness  adj fit  stag")
            self.logger.info("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                self.logger.info(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            self.logger.info('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        self.logger.info('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            self.logger.info("Generation time: {0:.3f} sec ({1:.3f} average)\n".format(elapsed, average))
        else:
            self.logger.info("Generation time: {0:.3f} sec\n".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        self.logger.info('****** Generation {0} Results ******'.format(self.generation))
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        self.logger.info('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        self.logger.info(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        self.logger.info('All species extinct.')

    def found_solution(self, config, generation, best):
        self.logger.info('Best individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            self.logger.info("Species {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        self.logger.info(msg)