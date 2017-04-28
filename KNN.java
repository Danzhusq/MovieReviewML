import java.util.ArrayList;

/**
 * Created by danny on 4/26/2017.
 */
public class KNN extends MachineLearningModel{
    private ArrayList<String> dictionary;
    private ArrayList<ArrayList<Integer>> trainingData;
    private ArrayList<ArrayList<Integer>> testData;
    private ArrayList<ArrayList<Double>> normalizedData;
    private boolean normalized;
    private boolean isManhattan;

    private int numOfNeihgbor;

    public KNN(ArrayList<String> dictionary, ArrayList<ArrayList<Integer>> trainingData,
               ArrayList<ArrayList<Integer>> testData, int numOfNeihgbor, boolean isManhattan) {
        this.dictionary = dictionary;
        this.trainingData = trainingData;
        this.testData = testData;
        this.numOfNeihgbor = numOfNeihgbor;

        this.normalizedData = new ArrayList<>();
        normalized = false;
        this.isManhattan = isManhattan;

        tPositive = -1;
        tPositive = -1;
        tNegative = -1;
        fNegative = -1;

        accuracy = -1;
        recall = -1;
        precision = -1;
    }

    @Override
    /**
     * No training required for KNN classifier.
     */
    public void train() {
    }

    /**
     *
     */
    @Override
    /**
     * Test function
     */
    public void test() {
        if(normalized) {
            testOriginal(testData);
        }
        else {
            testNormalized(normalizedData);
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

    public ArrayList<ArrayList<Integer>> getTestData() {
        return testData;
    }

    public void setTestData(ArrayList<ArrayList<Integer>> testData) {
        this.testData = testData;
    }

    public ArrayList<ArrayList<Double>> getNormalizedData() {
        return normalizedData;
    }

    public void setNormalizedData(ArrayList<ArrayList<Double>> normalizedData) {
        this.normalizedData = normalizedData;
    }

    public boolean isNormalized() {
        return normalized;
    }

    public void setNormalized(boolean normalized) {
        this.normalized = normalized;
    }

    public boolean isManhattan() {
        return isManhattan;
    }

    public void setManhattan(boolean manhattan) {
        isManhattan = manhattan;
    }

    public int getNumOfNeihgbor() {
        return numOfNeihgbor;
    }

    public void setNumOfNeihgbor(int numOfNeihgbor) {
        this.numOfNeihgbor = numOfNeihgbor;
    }


    //helper method


    /**
     * Test Original test data with integer input
     * @param test
     * Arraylist of test data of type integer to be labeled.
     */
    private void testOriginal(ArrayList<ArrayList<Integer>> test) {
        ArrayList<Integer> indexArray;
        //outer loop, label all the test data.
        for(int i = 0; i < test.size(); i++) {
             //inner loop, label the ith data.
            ArrayList<Double> distanceVector = computeDistance(trainingData, test.get(i), isManhattan);
            indexArray = findIndex(distanceVector, numOfNeihgbor);
            int actualType = i % 2 == 0 ? 1 : -1;
            int testType = label(indexArray);

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
            calculatePerformance();
        }
    }

    /**
     * Test Normalized test data with double input.
     * @param test
     * Arraylist of test data of type double to be labled.
     */
    private void testNormalized(ArrayList<ArrayList<Double>> test) {
        ArrayList<Integer> indexArray;
        //outer loop, label all the test data.
        for(int i = 0; i < test.size(); i++) {
            //inner loop, label the ith data.
            ArrayList<Double> distanceVector = computeDistanceNorm(trainingData, test.get(i),
                    isManhattan);
            indexArray = findIndex(distanceVector, numOfNeihgbor);
            int actualType = i % 2 == 0 ? 1 : -1;
            int testType = label(indexArray);

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
     * Normalize data, making it into data set with mean of 0 and unit variance on each feature.
     */
    private void normalizeTestData() {
        ArrayList<Double> meanArray = new ArrayList<>();
        ArrayList<Double> varianceArray = new ArrayList<>();

        double currentMean;
        double currentVariance;

        //calculate the mean and variance for the current feature first.
        for(int i = 0; i < testData.get(0).size(); i++) {
            currentMean = 0;
            currentVariance = 0;
            for( int j = 0; j < testData.size(); j++) {
                currentMean += testData.get(j).get(i);
            }
            currentMean = currentMean / (double)testData.size();
            for(int j = 0; j < testData.size(); j++) {
                currentVariance += Math.pow(testData.get(j).get(i) - currentMean, 2.0);
            }
            currentVariance = currentVariance / (double)testData.size();
            meanArray.add(currentMean);
            varianceArray.add(Math.sqrt(currentVariance));
        }

        //normalize the vector
        for(int i = 0; i < testData.size(); i++) {
            //normalize each vector
            ArrayList<Double> modifiedData = new ArrayList<>();
            for(int j = 0; j < testData.get(0).size(); j++) {
                double modifiedValue = (testData.get(i).get(j) - meanArray.get(j)) /
                         varianceArray.get(j);
                modifiedData.add((modifiedValue));
            }
            normalizedData.add(modifiedData);
        }
    }

    /**
     * Label the data based on the index array
     * @param indexArray
     * array contains the index of its k closest neighbors.
     * @return
     * Label as positive, 1, if result is actually positive, label as negative, -1, if is actually negative.
     */
    private int label(ArrayList<Integer> indexArray) {
        int result = 0;

        int posCount = 0;
        int negCount = 0;

        for(int i = 0; i < indexArray.size(); i++) {
            if(indexArray.get(i) % 2 == 0) {
                posCount++;
            }
            else {
                negCount++;
            }
        }
        result = posCount > negCount ? 1 : -1;
        return result;
    }

    /**
     * Compute the distance of the given data and the training sample data set.
     * @param sample
     * sample data set.
     * @param test
     * test data to be labeled, origanl data required, list of integers.
     * @param isManhattan
     * indicate if manhattan distance is requested.
     * @return
     * Arraylist of doubles indicating distance between the test data and the sample set.
     */
    private ArrayList<Double> computeDistance(ArrayList<ArrayList<Integer>> sample,
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

    /**
     * Compute the distance of the given data and the training sample data set.
     * @param sample
     * sample data set.
     * @param test
     * test data to be labeled, normalized data required, arraylist of doubles.
     * @param isManhattan
     * indicate if manhattan distance is requested.
     * @return
     * Arraylist of doubles indicating distance between the test data and the sample set.
     */
    private ArrayList<Double> computeDistanceNorm(ArrayList<ArrayList<Integer>> sample,
                                              ArrayList<Double> test, boolean isManhattan) {
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

    /**
     * Find the index array of the k nearest point as requested.
     * @param distance
     * distance array of the test data and the sample set.
     * @param numOfNeihgbor'
     * number of neighbors required.
     * @return
     * Arraylist of integers, which are the index of its closest k neighbors.
     */
    private ArrayList<Integer> findIndex(ArrayList<Double> distance, int numOfNeihgbor) {
        ArrayList<Integer> result = new ArrayList<>();
        double currentMin = distance.get(0);
        int currentMinIndex = 0;
        while(result.size() < numOfNeihgbor) {
            for(int i = 0; i < distance.size(); i++) {
                if(distance.get(i) < currentMin && !result.contains(i)) {
                    currentMin = distance.get(i);
                    currentMinIndex = i;
                }
            }
            result.add(currentMinIndex);
        }

        return result;
    }
}
