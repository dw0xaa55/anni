#define ANNI_IMPLEMENTATION
#include "anni.h"

int main(){
  srand(time(NULL));

  // network topology
  size_t topology[] = {2, 3, 1}; 
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
  double ANNI_learning_rate = 15.0;
  double ANNI_epochs        = 5000;

  printf("\033[2J");
  double mse = 0.0;
  for(size_t i = 0; i <= ANNI_epochs; ++i){
    size_t sample = i % SAMPLE_AMOUNT; // change
    train(nn, input[sample], output[sample], ANNI_learning_rate);
    printTraining(nn, i); // use for debugging only
    
    // error evaluation (MSE)  [fixme::what if there are more than output neurons]
    mse += pow((nn->neurons[NUM_LAYERS - 1][0] - output[sample][0]), 2);
    if(sample == SAMPLE_AMOUNT-1){
      mse /= SAMPLE_AMOUNT;
      printf("\033[36mError: \033[0m %f\n", mse);
      mse = 0.0;
    }
  }
  printf("\n");

  // network check
  printf("\n\033[33mtrained network:\033[0m\n");
  for (int i = 0; i < SAMPLE_AMOUNT; i++){
    feedForward(nn, input[i]);
    printf("Input: [%d, %d] -> Output: ", (int)input[i][0], (int)input[i][1]);
    printOutput(nn);
  }

  // sanity checks
  printf("%zu", NUM_LAYERS);

  freeNeuralNetwork(nn);
  return 0;
}
