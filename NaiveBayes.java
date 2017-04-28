import javax.swing.plaf.nimbus.NimbusLookAndFeel;
import java.util.ArrayList;

/**
 * Created by danny on 4/26/2017.
 */
public class NaiveBayes extends MachineLearningModel{
    private ArrayList<String> dictionary;
    private ArrayList<ArrayList<Integer>> trainingData;
    private ArrayList<ArrayList<Double>> probability;
    private ArrayList<ArrayList<Integer>> testData;


    public NaiveBayes(ArrayList<String> dictionary, ArrayList<ArrayList<Integer>> trainingData) {
        this.dictionary = dictionary;
        this.trainingData = trainingData;
        probability = new ArrayList<ArrayList<Double>>();
        testData = new ArrayList<>();

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
     * Training classifier using training data set to get likelihood vector of different class label.
     */
    @Override
    public void train() {
        int posCount;
        int negCount;

        double virtualProbablity = ( 0.0 + ( 16 * 1.0 / 2 )) / (800 + 16);

        ArrayList<Double> posProbablityList = new ArrayList<>();
        ArrayList<Double> negProbablityList = new ArrayList<>();

        //computer probalitity for each feature, p(wi|C)
        for( int i = 0; i < trainingData.get(0).size(); i++) {
            posCount = 0;
            negCount = 0;
            for( int j = 0; j < trainingData.size(); j++) {
                if( j % 2 == 0) {
                    posCount += trainingData.get(j).get(i);
                }
                else {
                    negCount += trainingData.get(j).get(i);
                }
            }
            //check if any of it is 0
            posProbablityList.add(posCount != 0 ? posCount / 800.0 : virtualProbablity);
            negProbablityList.add(negCount != 0 ? negCount / 800.0 : virtualProbablity);
        }
        probability.add(posProbablityList);
        probability.add(negProbablityList);
    }

    @Override
    /**
     * Test data using provided test data set and calculate stats.
     */
    public void test() {
        int actualType;
        int testType;
        for( int i = 0; i < testData.size(); i++) {
            actualType = i % 2;
            testType = label(testData.get(i));
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

    public ArrayList<String> getDictionary() {
        return dictionary;
    }

    public void setDictionary(ArrayList<String> dictionary) {
        this.dictionary = dictionary;
    }

    public ArrayList<ArrayList<Integer>> getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(ArrayList<ArrayList<Integer>> trainingData) {
        this.trainingData = trainingData;
    }

    public ArrayList<ArrayList<Double>> getProbability() {
        return probability;
    }

    public void setProbability(ArrayList<ArrayList<Double>> probability) {
        this.probability = probability;
    }

    public ArrayList<ArrayList<Integer>> getTestData() {
        return testData;
    }

    public void setTestData(ArrayList<ArrayList<Integer>> testData) {
        this.testData = testData;
    }


    //helper method

    /**
     * Private helper method to label the testData based on the learned proability vector.
     * @param data
     * Input Integer Vector
     * @return
     * Label as positive, 1, if result is actually positive, label as negative, -1, if is actually negative.
     */
    private int label(ArrayList<Integer> data) {
        double posPrior = 0.5;
        double negPrior = 0.5;

        double posTemp = posPrior;
        double negTemp = negPrior;
        for( int i = 0; i < data.size(); i++) {
            posTemp *= probability.get(0).get(i);
            negTemp *= probability.get(1).get(i);
        }
        return posTemp > negTemp ? 1 : -1;
    }
}
