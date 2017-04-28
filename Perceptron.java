import java.util.ArrayList;

/**
 * Created by danny on 4/26/2017.
 */
public class Perceptron extends MachineLearningModel {
    private double learningRate;

    private ArrayList<String> dictionary;
    private ArrayList<Double> weight;

    private ArrayList<ArrayList<Integer>> trainingData;
    private ArrayList<ArrayList<Integer>> testData;


    /**
     * Public constructor for a given dictionary and training data.
     * @param dictionary
     * @param trainingData
     */
    public Perceptron(ArrayList<String> dictionary, ArrayList<ArrayList<Integer>> trainingData,
                      ArrayList<ArrayList<Integer>> testData) {
        this.dictionary = dictionary;
        this.trainingData = trainingData;
        weight = new ArrayList<>();
        this.testData = testData;

        learningRate = Math.random();

        tPositive = -1;
        tPositive = -1;
        tNegative = -1;
        fNegative = -1;

        accuracy = -1;
        recall = -1;
        precision = -1;
    }


    //functionality

    /**
     * Trains the perceptron weight vector based on the training data.
     */
    @Override
    public void train() {
        initializeWeight();

        //maximum training iterations if not completed yet.
        int iteration = 0;
        int maxIteration = 100;
        //indicate if adaptation is required.
        boolean errorFound = true;
        while (errorFound && iteration < maxIteration) {
            errorFound = false;
            for( int i = 0; i < trainingData.size(); i++) {
                int error = (i % 2 == 0 ? 1 : -1) - label(weight, trainingData.get(i));
                errorFound = error != 0;
                if(errorFound) {
                    for( int j = 0; j < trainingData.get(i).size(); j++) {
                        double newWeight = weight.get(j) + learningRate * error * trainingData
                                .get(i).get(j);
                        weight.set(j, newWeight);
                    }
                }
            }
        }
    }

    /**
     * Test the data to see the performance.
     */
    @Override
    public void test() {
        for(int i = 0; i < testData.size(); i++) {
            int actualType = i % 2 == 0 ? 1 : -1;
            int testType = label(weight, testData.get(i));
            if (actualType == 1) {
                if(testType == 1) {
                    tPositive++;
                }
                else {
                    fPositive++;
                }
            }
            else {
                if(testType == -1) {
                    tNegative++;
                }
                else {
                    fNegative++;
                }
            }
        }
        calculatePerformance();
    }

    /**
     * Reset perceptron to a fully untrained perceptron.
     */
    public void reset() {
        dictionary = new ArrayList<>();
        trainingData = new ArrayList<>();

        learningRate = Math.random();

        weight = new ArrayList<>();
        testData = new ArrayList<>();
    }

    /**
     * update perceptron with new set of dictionary data and trainingData. All parameters are reset.
     * @param dictionary
     * new dictionary file to train.
     * @param trainingData
     * new training data.
     */
    public void update(ArrayList<String> dictionary, ArrayList<ArrayList<Integer>> trainingData) {
        this.dictionary = dictionary;
        this.trainingData = trainingData;
        learningRate = Math.random();
        weight = new ArrayList<>();
        testData = new ArrayList<>();

        tPositive = -1;
        tPositive = -1;
        tNegative = -1;
        fNegative = -1;

        accuracy = -1;
        recall = -1;
        precision = -1;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public ArrayList<String> getDictionary() {
        return dictionary;
    }

    public void setDictionary(ArrayList<String> dictionary) {
        this.dictionary = dictionary;
    }

    public ArrayList<Double> getWeight() {
        return weight;
    }

    public void setWeight(ArrayList<Double> weight) {
        this.weight = weight;
    }

    public ArrayList<ArrayList<Integer>> getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(ArrayList<ArrayList<Integer>> trainingData) {
        this.trainingData = trainingData;
    }

    public ArrayList<ArrayList<Integer>> getTestData() {
        return testData;
    }

    public void setTestData(ArrayList<ArrayList<Integer>> testData) {
        this.testData = testData;
    }


    //helper method

    /**
     * Initialize weight according to the length of the dictionary, making sure it matches.
     */
    private void initializeWeight() {
        for(int i = 0; i < dictionary.size(); i++) {
            weight.add(Math.random());
        }
    }

    /**
     * Label a data based on current weight, if positive, label 1, as positive, otherwise, -1 as negative.
     * @param weight
     * weight vector
     * @param data
     * data to be labeled.
     * @return
     * Label as positive, 1, if result is actually positive, label as negative, -1, if is actually negative.
     */
    private int label(ArrayList<Double> weight, ArrayList<Integer> data) {
        double result = 0;
        for(int i = 0; i < weight.size(); i++) {
            result += weight.get(i)*data.get(i);
        }
        return result > 0 ? 1 : -1;
    }

}
