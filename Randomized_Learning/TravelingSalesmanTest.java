import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import java.util.Arrays;
import java.util.Random;

/**
 * Adapted code from https://github.com/pushkar/ABAGAIL (the ABAGAIL test repository)
 */
public class TravelingSalesmanTest {

    public static void main(String[] args) {
        int N = Integer.parseInt(args[0]);
        Random random = new Random();
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        int iterations = Integer.parseInt(args[1]);
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

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
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }

        System.out.println("Genetic Algorithm");
        for(int i = 0; i < iterations; i++)
        {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 10, gap);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }

        System.out.println("MIMIC");
        int[] distr = new int[N];
        Arrays.fill(distr, N);
        odd = new  DiscreteUniformDistribution(distr);
        Distribution df = new DiscreteDependencyTree(.1, distr;
        for(int i = 0; i < iterations; i++)
        {
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            MIMIC mimic = new MIMIC(200, 60, pop);
            long currT = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - currT))/ 1e9d));
        }
    }
}
