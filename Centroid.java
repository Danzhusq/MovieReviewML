import java.util.ArrayList;

/**
 * Created by danny on 4/26/2017.
 */
public class Centroid extends MachineLearningModel{
    private ArrayList<String> dictionary;

    private ArrayList<ArrayList<Integer>> trainingData;
    private ArrayList<ArrayList<Integer>> testData;

    private boolean isManhattan;

    private ArrayList<Double> posCenter;
    private ArrayList<Double> negCenter;

    public Centroid(ArrayList<String> dictionary, ArrayList<ArrayList<Integer>> trainingData,
                    ArrayList<ArrayList<Integer>> testData, boolean isManhattan) {
        this.dictionary = dictionary;
        this. trainingData = trainingData;
        this.testData = testData;
        this.isManhattan = isManhattan;

        posCenter = new ArrayList<>();
        negCenter = new ArrayList<>();
    }


    @Override
    /**
     * Generate the center for each feature with respect to positive and negative reviews.
     */
    public void train() {
        for(int i = 0; i < trainingData.get(0).size(); i++) {
            double currentPosCenter = 0;
            double currentNegCenter = 0;
            for(int j = 0; j < trainingData.size();) {
                currentPosCenter += trainingData.get(j++).get(i);
                currentNegCenter += trainingData.get(j++).get(i);
            }
            posCenter.add(currentPosCenter / 800.0);
            negCenter.add(currentNegCenter / 800.0);
        }
    }

    @Override
    /**
     * Test function.
     */
    public void test() {
        for (int i = 0; i < testData.size(); i++) {
            int actualType = i % 2 == 0 ? 1 : -1;
            int testType = label(testData.get(i));

            if (actualType == 1) {
                if (testType == 1) {
                    tPositive++;
                } else {
                    fPositive++;
                }
            } else {
                if (testType == -1) {
                    tNegative++;
                } else {
                    fNegative++;
                }
            }
        }
        calculatePerformance();
    }









    //helper function

    /**
     * Label current test Data with 1 or -1.
     * @param test
     * test data to be labeled.
     * @return
     * label as positive, 1, if it's closer to positive center, negative, -1, otherwise.
     */
    private int label(ArrayList<Integer> test) {
        ArrayList<ArrayList<Double>> sampleCenter = new ArrayList<>();
        sampleCenter.add(posCenter);
        sampleCenter.add(negCenter);
        ArrayList<Double> distance = computeDistance(sampleCenter, test, isManhattan);
        return distance.get(0) > distance.get(1) ? 1 : -1;
    }


    /**
     * Compute the distance of the given data and the training sample data set.
     * @param sample
     * sample data set.
     * @param test
     * test data to be labeled.
     * @param isManhattan
     * indicate if manhattan distance is requested.
     * @return
     * Arraylist of doubles indicating distance between the test data and the sample set.
     */
    private ArrayList<Double> computeDistance(ArrayList<ArrayList<Double>> sample,
                                              ArrayList<Integer> test, boolean isManhattan) {
        ArrayList<Double> result = new ArrayList<>();
        //iterate through the list
        for(int i = 0; i < sample.size(); i++) {
            //compute distance for each data.
            double distance = 0;
            for(int j = 0; j < sample.get(0).size(); j++) {
                if(isManhattan) {
                    distance += Math.abs((double)(sample.get(i).get(j)) - test.get(j));
                }
                else {
                    distance += Math.sqrt(Math.pow((double)(sample.get(i).get(j)) - test.get(j),
                            2.0));
                }
            }
            result.add(distance);
        }
        return result;
    }
}
