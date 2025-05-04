#define ANNI_IMPLEMENTATION
#include "anni.h"

int main(){
  srand(time(NULL));

  // network topology
  size_t topology[] = {2, 3, 2, 1}; 
#define NUM_LAYERS (sizeof(topology) / sizeof(topology[0]))

  // initialize neural network.
  NeuralNetwork *nn = createNeuralNetwork(topology, NUM_LAYERS);

  // training data
#define SAMPLE_AMOUNT 4
  double input[SAMPLE_AMOUNT][2] = {{0.0, 0.0},
				    {0.0, 1.0},
				    {1.0, 0.0},
				    {1.0, 1.0}};
    
  double output[SAMPLE_AMOUNT][1] = {{0.0},
				     {1.0},
				     {1.0},
				     {0.0}};

  // network training
  double ANNI_learning_rate = 0.1;
  double ANNI_epochs        = 1000000;


  for (int i = 0; i < ANNI_epochs; ++i){
    size_t sample = i % SAMPLE_AMOUNT; // change
    train(nn, input[sample], output[sample], ANNI_learning_rate);
  }

  // network check
  printf("trained network:\n");
  for (int i = 0; i < SAMPLE_AMOUNT; i++){
    feedForward(nn, input[i]);
    printf("Input: [%d, %d] -> Output: ", (int)input[i][0], (int)input[i][1]);
    printOutput(nn);
  }

  freeNeuralNetwork(nn);
  return 0;
}
