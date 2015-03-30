import java.io.*;
import java.util.*;

public class NearestNeighbor {
    private ArrayList<ArrayList<Double>> database;
    private ArrayList<String> labels;

    /**
     * Constructor
     */
    NearestNeighbor() {
        database = null;
        labels = null;
    }

    /**
     * Train the classifier using data stored in trainfilename.
     *
     * @param trainfilename name of the training data set file
     */
    public void train(String trainfilename)
            throws FileNotFoundException {

        System.out.println("Training on " + trainfilename);

        // Clear the current data base
        database = new ArrayList<ArrayList<Double>>();
        labels = new ArrayList<String>();

        File trainFile = new File(trainfilename);
        Scanner input = new Scanner(trainFile);
        // Read in the training data set
        while (input.hasNextLine()) {
            String line = input.nextLine();
            Scanner instance = new Scanner(line).useDelimiter(",");
            // Label comes first (String)
            String l = instance.next();

            //----- Store this label in labels -----//
            labels.add(l);

            // Then the features
            ArrayList<Double> features = new ArrayList<Double>();
            while (instance.hasNext())
                features.add(instance.nextDouble());
            //----- Store the feature vector in the database -----//
            database.add(features);
        }

        System.out.println("Stored " + database.size() + " instances.");
        System.out.println(labels);
        System.out.println(database);
    }

    /**
     * Classify a single instance by finding its nearest neighbor.
     *
     * @param instance the feature vector to be classified
     * @return classification (label) of the new instance
     */
    public String classify(ArrayList<Double> instance) {
        int nearest = 0;     //----- update these ------//
        double minDist = Integer.MAX_VALUE;  //----- update these -----//

        //----- Find the nearest neighbor ------//
        // (You may want to implement the Euclidean distance
        // calculation as a separate method)
        double cur;
        for (int i = 0; i < database.size(); i++) {
            cur = calculateDist(instance, database.get(i));
            if (cur < minDist) {
                minDist = cur;
                nearest = i;
            }
        }

        // Report the nearest neighbor
        System.out.print("The nearest neighbor to " + instance);
        System.out.printf(" is item %d, distance %.2f.\n", nearest, minDist);

        // Return the nearest neighbor's label
        return labels.get(nearest);
    }

    private double calculateDist(List<Double> aList, List<Double> bList) {
        double cum = 0, res;

        for (int i = 0; i < aList.size(); i++) {
            res = aList.get(i) - bList.get(i);
            cum += res*res;
        }
        return Math.sqrt(cum);
    }

    /**
     * Test the classifier using data stored in testfilename.
     *
     * @param testfilename name of the testing data set file
     * @return average accuracy
     */
    public double test(String testfilename)
            throws FileNotFoundException {
        //----- Create local variables to store the test data -----//
        // (do not add them to the training data!)

        System.out.println("Testing on " + testfilename);

        int total = 0, good = 0, bad = 0;


        File testFile = new File(testfilename);
        Scanner input = new Scanner(testFile);
        // Read in the testing data set
        while (input.hasNextLine()) {
            String line = input.nextLine();
            Scanner instance = new Scanner(line).useDelimiter(",");
            // Label comes first (String)
            String l = instance.next();
            //----- Store this label -----//

            // Then the features
            ArrayList<Double> features = new ArrayList<Double>();
            while (instance.hasNext()) {
                features.add(instance.nextDouble());
            }
            //----- Store this feature vector -----//
            total++;
            String l2 = classify(features);
            if (l2.equals(l)) {
                good++;
            } else {
                bad++;
            }
        }

        //----- Report how many test instances were read in. -----//
        System.out.println("Total: " + total + "\n Good: " + good + "\n Bad: " + bad);

        //----- Classify each of the test items, compare the output


        // to their true labels, and report average accuracy across the test set.
        double acc = (double) good / total;

        return acc;
    }

    /**
     * Main method.
     *
     * @param args training and test file names.
     */
    public static void main(String args[]) {
        if (args.length != 2) {
            System.err.println("Error: specify training and test file names.");
            System.exit(1);
        }

        // Train the classifier
        NearestNeighbor classifier = new NearestNeighbor();
        try {
            classifier.train(args[0]);
        } catch (FileNotFoundException e) {
            System.out.println("Ooops, " + args[0] + " doesn't exist!");
        }

        // Test the classifier
        try {
            System.out.println(classifier.test(args[1]));
        } catch (FileNotFoundException e) {
            System.out.println("Ooops, " + args[1] + " doesn't exist!");
        }
    }


}
