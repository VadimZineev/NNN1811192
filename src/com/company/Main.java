package com.company;

public class Main {

    public static double random(double minRandom, double maxRandom) {
        double randomNumber = Math.random() * maxRandom + minRandom;
        return randomNumber;
    }

    public static double sigmoid(double x) {
        x = 1 / (1 + Math.pow(Math.exp(x), -x));
        return x;
    }

    public static void main(String[] args) {
        //Parameters
        String[] info = new String[8];
        double[] expected = {0, 1, 0, 0, 1, 1, 0, 0}; //ожидаемый ответ
        double actual = 0; //полученный ответ
        double sigmdx = 0;
        double error = 0;
        double weightsDelta = 0;
        double learningRate = 0.1;
        int epoch = 0;

        boolean positive = true;
        boolean negative = true;

        //Input layout
        double[][] threeStartNeuron = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

        double[] weightsFromStartNeuron1 = new double[2];
        double[] weightsFromStartNeuron2 = new double[2];
        double[] weightsFromStartNeuron3 = new double[2];

        //Hidden layout
        double[] twoHiddenNeuron = new double[2];
        double[] weightsFromHiddenNeurons = new double[2];

        //Output layout
        double oneOutputNeuronAfterSig = 0;

        //Заполнение весов
        for (int i = 0; i < 2; i++) {
            weightsFromStartNeuron1[i] = random(0.1, 0.9);
            weightsFromStartNeuron2[i] = random(0.1, 0.9);
            weightsFromStartNeuron3[i] = random(0.1, 0.9);
            weightsFromHiddenNeurons[i] = random(0.1, 0.9);
        }

        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < 2; j++) {
                twoHiddenNeuron[j] = threeStartNeuron[i][0] * weightsFromStartNeuron1[j] + threeStartNeuron[i][1] * weightsFromStartNeuron2[j] +
                        +threeStartNeuron[i][2] * weightsFromStartNeuron3[j];
                oneOutputNeuronAfterSig += sigmoid(twoHiddenNeuron[j]) * weightsFromHiddenNeurons[j];
            }
            actual = sigmoid(oneOutputNeuronAfterSig);
            error = actual - expected[i];
            sigmdx = actual * (1 - actual);
            weightsDelta = error * sigmdx;

            for (int k = 0; k < 8; k++) {
                if (expected[k] == 0) {
                    while (negative) {
                        for (int j = 0; j < 2; j++) {
                            weightsFromHiddenNeurons[j] = (weightsFromHiddenNeurons[j] - sigmoid(twoHiddenNeuron[j])) * weightsDelta * learningRate;
                            weightsFromStartNeuron1[j] = (weightsFromStartNeuron1[j] - threeStartNeuron[k][j]) * weightsDelta * learningRate;
                            weightsFromStartNeuron2[j] = (weightsFromStartNeuron2[j] - threeStartNeuron[k][j]) * weightsDelta * learningRate;
                            weightsFromStartNeuron3[j] = (weightsFromStartNeuron3[j] - threeStartNeuron[k][j]) * weightsDelta * learningRate;
                            twoHiddenNeuron[j] = threeStartNeuron[k][0] * weightsFromStartNeuron1[j] + threeStartNeuron[k][1] * weightsFromStartNeuron2[j] +
                                    +threeStartNeuron[k][2] * weightsFromStartNeuron3[j];
                            oneOutputNeuronAfterSig += sigmoid(twoHiddenNeuron[j]) * weightsFromHiddenNeurons[j];
                        }

                        actual = sigmoid(oneOutputNeuronAfterSig);
                        if (oneOutputNeuronAfterSig > 0.1 && oneOutputNeuronAfterSig < 0.2) {
                            info[k] = "Epoch: " + epoch + " | " + "Result: " + oneOutputNeuronAfterSig + " | " + "Error: " + error;
                            System.out.println(info[k]);
                            break;
                        } else {
                            ++epoch;
                            error = actual - expected[i];
                            sigmdx = actual * (1 - actual);
                            weightsDelta = error * sigmdx;
                        }
                    }
                }
                if (expected[k] == 1) {
                    while (positive) {
                        for (int j = 0; j < 2; j++) {
                            weightsFromHiddenNeurons[j] = (weightsFromHiddenNeurons[j] + sigmoid(twoHiddenNeuron[j])) * weightsDelta * learningRate;
                            weightsFromStartNeuron1[j] = (weightsFromStartNeuron1[j] + threeStartNeuron[k][j]) * weightsDelta * learningRate;
                            weightsFromStartNeuron2[j] = (weightsFromStartNeuron2[j] + threeStartNeuron[k][j]) * weightsDelta * learningRate;
                            weightsFromStartNeuron3[j] = (weightsFromStartNeuron3[j] + threeStartNeuron[k][j]) * weightsDelta * learningRate;
                            twoHiddenNeuron[j] = threeStartNeuron[k][0] * weightsFromStartNeuron1[j] + threeStartNeuron[k][1] * weightsFromStartNeuron2[j] +
                                    +threeStartNeuron[k][2] * weightsFromStartNeuron3[j];
                            oneOutputNeuronAfterSig += sigmoid(twoHiddenNeuron[j]) * weightsFromHiddenNeurons[j];
                        }

                        actual = sigmoid(oneOutputNeuronAfterSig);

                        if (oneOutputNeuronAfterSig > 0.9 && oneOutputNeuronAfterSig < 1.0) {
                            info[k] = "Epoch: " + epoch + " | " + "Result: " + oneOutputNeuronAfterSig + " | " + "Error: " + error;
                            System.out.println(info[k]);
                            break;
                        } else {
                            ++epoch;
                            error = actual - expected[i];
                            sigmdx = actual * (1 - actual);
                            weightsDelta = error * sigmdx;
                        }
                    }
                }
            }
        }
    }
}

