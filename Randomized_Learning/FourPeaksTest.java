import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;
import java.util.Arrays;

/**
 * Adapted code from https://github.com/pushkar/ABAGAIL (the ABAGAIL test repository)
 */
public class FourPeaksTest {

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        int T = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("Randomized Hill Climbing");
        for(int i = 0; i < iterations; i++)
        {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            System.out.println(ef.value(rhc.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }

        System.out.println("Simulated Annealing");
        for(int i = 0; i < iterations; i++)
        {
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }

        System.out.println("Genetic Algorithm");
        for(int i = 0; i < iterations; i++)
        {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }

        System.out.println("MIMIC");
        for(int i = 0; i < iterations; i++)
        {
            MIMIC mimic = new MIMIC(200, 5, pop);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }
    }
}
