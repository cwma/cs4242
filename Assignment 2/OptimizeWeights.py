import subprocess
import sys
import random
import json
import multiprocessing
import Tweet
import operator
from collections import Counter
from queue import Queue
from sklearn import metrics

class ClassifierWorker(multiprocessing.Process):
    
    def __init__(self, event, queue, results):
        train_dataset, test_dataset = Tweet.get_flattened_data('dataset/k4/training.json', 'dataset/k4/testing.json', 'dataset/k4/root_tweet.json', 4)
        multiprocessing.Process.__init__(self)
        self._event = event
        self._queue = queue
        self._results = results
        self.daemon = True
        self._tag_count = 2
        self._preprocesssed_results = [json.load(open("dataset/NaiveBayesCascadeClassifier_results.json", 'r')),
                                       json.load(open("dataset/RandomForestCascadeClassifier_results.json", 'r')),
                                       json.load(open("dataset/knnClassifier_results.json", 'r')),
                                       json.load(open("dataset/SvmCascadeClassifier_results.json", 'r'))]
        self._cascades = test_dataset

    def _combine_scores(self, scores, weights):
        final_score = {"positive": 0, "negative": 0}
        score_and_weight = zip(scores, [weights[i:i+self._tag_count] for i in range(0,len(self._preprocesssed_results)*self._tag_count,self._tag_count)])
        for score, (pos_weight, neg_weight) in score_and_weight:
            final_score["positive"] += score["positive"] * pos_weight
            final_score["negative"] += score["negative"] * neg_weight
        return final_score

    def _classify(self, weights):
        results = {"prediction": [], "actual": []}
        for cascade in self._cascades:
            scores = [results[cascade['url']] for results in self._preprocesssed_results]
            final_score = self._combine_scores(scores, weights)
            final_score = sorted(final_score.items(), key=operator.itemgetter(1))
            result = final_score[-1][0]
            results["prediction"].append(result)
            if cascade['label']:
                results["actual"].append("positive")
            else:
                results["actual"].append("negative")
        return metrics.f1_score(results['actual'], results['prediction'], average="micro")

    def run(self):
        """run method"""
        while self._event.is_set():
            index, weights = self._queue.get()
            result = self._classify(weights)
            self._results.put((index, result))
            self._queue.task_done()

class OptimizeWeights():
    """
    Genetic Algorithm to optimise weights for Cascade classification
    """

    #_NUM_WORKERS = multiprocessing.cpu_count()
    _NUM_WORKERS = 4
    population_size = 10000 # number of agents
    selection = 0.1 # random pool size to select best parents from
    culling = 0.3 # % of population to cull and replace every generation
    mutation_rate = 0.2 # mutation rate
    mutation_delta = 0.5 # % range of mutation adjustment
    num_labels = 2
    num_classifiers = 4
    num_weights = num_classifiers * num_labels # no of classifiers * labels

    def __init__(self):
        """
        initializes the multiprocessing data structures and spawn workers
        """
        self._event = multiprocessing.Event()
        self._queue = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._spawn_workers()
        self.population = self._seed_population()

    def _spawn_workers(self):
        """spawns the threaded worker class instances"""
        self._event.set()
        self._workers = [ClassifierWorker(self._event, self._queue, self._results) for x in range(self._NUM_WORKERS)]
        [worker.start() for worker in self._workers]

    def _queue_search(self, population):
        """puts the work into the queue for the worker instances to consume"""
        [self._queue.put(p) for p in enumerate(population)]

    def _normalize_fitness(self, fitness):
        """normalize fitness"""
        sum_fitness = sum(fitness)
        return [sum_fitness > 0 and (float(w) / sum_fitness) or 0.5 for w in fitness]

    def _normalize_weights(self, weights):
        """normalize all weight group values to 1. if all weights are 0 return 0.5 (for crossover average weighted fitness)"""
        # turns [1,2,3,1,2,3] to [[1,1],[2,2],[3,3]] etc and each group must be normalized to 1
        sub_weight_groups = [[weights[x] for x in range(i,self.num_weights,self.num_labels)] for i in range(int(len(weights)/self.num_classifiers))]
        sum_weight_groups = [sum(map(abs, sub_weight)) for sub_weight in sub_weight_groups]
        weights = [[sum_weights > 0 and (float(w) / sum_weights) or 0.5 for w in weights] for weights, sum_weights in zip(sub_weight_groups, sum_weight_groups)]
        return [weight for sub_weights in [[weight[i] for weight in weights] for i in range(self.num_classifiers)] for weight in sub_weights]

    def _generate_weights(self):
        """generates a random vector of length num_weights that sums to 1.0"""
        weights = [random.uniform(0, 1) for x in range(self.num_weights)]
        return self._normalize_weights(weights)

    def _seed_population(self):
        """generates the initial population"""
        return [self._generate_weights() for x in range(self.population_size)]

    def _select_parents(self):
        """tournament selection"""
        random_selection = random.sample(range(self.population_size), int(self.population_size * 0.1))
        return sorted(random_selection, key=lambda s: self._scores.get(s), reverse=True)[:2]

    def _crossover(self, parent1, parent2):
        """average weighted crossover"""
        fitness1, fitness2 = self._normalize_fitness([self._scores[parent1], self._scores[parent2]])
        return self._normalize_weights([(fitness1 * p1) + (fitness2 * p2) for p1, p2 in zip(self.population[parent1], self.population[parent2])])

    def _mutate(self, offspring):
        """mutate randomly selected weight by delta and normalize"""
        weight_idx = random.choice(range(len(offspring)))
        mutation_modifier = 1 + random.uniform(-self.mutation_delta, self.mutation_delta)
        offspring[weight_idx] *= mutation_modifier
        return self._normalize_weights(offspring)

    def _create_offspring(self):
        """create an offspring using tournament selection and average weighted crossover"""
        parents = self._select_parents()
        offspring = self._crossover(*parents)
        if (random.uniform(0, 1) < self.mutation_rate):
            offspring = self._mutate(offspring)
        return offspring
        
    def _next_generation(self, ranks):
        """cull the weakest population and replace them with new offspring"""
        replace = ranks[:int(self.population_size * self.culling)]
        for idx in replace:
            self.population[idx] = self._create_offspring()

    def _report(self, ranks):
        """prints top 10 weights"""
        top10 = ranks[self.population_size-10:]
        for idx in top10[::-1]:
            print("F1: %s, Weights: %s" % (self._scores[idx], self.population[idx]))
        print("Population Average F1: %s" % (self._pop_f1 / float(self.population_size)))

    def optimize_weights(self, generations):
        """
        feeds the worker instances weights created every new generation
        and then consumes the results once all the workers have completed.
        """
        for gen in range(generations):
            print(" Generation: %s" % gen)
            self._pop_f1 = 0
            self._queue_search(self.population)
            self._queue.join()
            self._scores = {}
            while not self._results.empty():
                (index, f1) = self._results.get()
                self._scores[index] = f1
                self._pop_f1 += f1
            ranks = sorted(range(self.population_size), key=lambda s: (self._scores.get(s)))
            self._report(ranks)
            self._next_generation(ranks)

if __name__ == '__main__':
    ow = OptimizeWeights()
    ow.optimize_weights(10) 