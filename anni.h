/*
 * Anni - Artificial Neural Network Intelligence
 * Author : C. Huffenbach
 * Date   : May 2025
 * Version: 4.3
 *
 * TODO:
 *   - [X] print net
 *   - [ ] save net to file
 *   - [ ] load net from file
 *   - [ ] load trainingdata from file
 */

#ifndef _ANNI_H_
#define _ANNI_H_

// includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
  // this structure represents the Neural Network
  size_t num_layers;    // total number of layers (input + hidden + output)
  size_t *topology;     // array that defines how many neurons per layer
  double **neurons;     // array of neuron activations for each layer
  double **biases;      // biases for neurons in each layer (except the input layer)
  double ***weights;    // weight matrices: weights[l][i][j] connects j-th neuron in (l-1)th layer to i-th neuron in layer l
} NeuralNetwork;

#define RANDOM_WEIGHT() ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) // Macro to help initialize weights randomly between -1.0 and 1.0.

double         sigmoid(double x);
double         sigmoidDerivative(double output);
NeuralNetwork* createNeuralNetwork(const size_t *topology, size_t num_layers);
void           feedForward(NeuralNetwork *nn, double *input);
void           printOutput(NeuralNetwork *nn);
void           backPropagation(NeuralNetwork *nn, double *output, double learning_rate);
double         meanSquareErrorDerivative(double predicted, double target, size_t n);
void           train(NeuralNetwork *nn, double *input, double *output, double learning_rate);
void           freeNeuralNetwork(NeuralNetwork *nn);
void           printNetwork(NeuralNetwork *nn);
void           printTraining(NeuralNetwork *nn, size_t epoch);
// TODO
void           saveNetworkToFile(NeuralNetwork *nn, const char* filename);
void           loadNetworkFromFile(NeuralNetwork *nn, const char* filename);
#endif

#ifdef ANNI_IMPLEMENTATION
double sigmoid(double x)                { return 1.0 / (1.0 + exp(-x));   }
double sigmoidDerivative(double output) { return output * (1.0 - output); }

NeuralNetwork*
createNeuralNetwork(const size_t *topology, size_t num_layers){
  // create and initialize the neural network given the topology array and number of layers.
  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  nn->num_layers = num_layers;
  nn->topology = malloc(num_layers * sizeof(size_t));
  for(size_t i = 0; i < num_layers; ++i)
    nn->topology[i] = topology[i];
    
  // allocate memory for neuron activations (one array per layer)
  nn->neurons = malloc(num_layers * sizeof(double*));
  for(size_t i = 0; i < num_layers; ++i)
    nn->neurons[i] = calloc(topology[i], sizeof(double));
    
  // allocate and initialize biases (starting from layer 1 as the input layer has no biases)
  nn->biases = malloc(num_layers * sizeof(double*));
  nn->biases[0] = NULL;  // Input layer
  for(size_t i = 1; i < num_layers; ++i){
    nn->biases[i] = malloc(topology[i] * sizeof(double));
    for(size_t j = 0; j < topology[i]; ++j){
      nn->biases[i][j] = RANDOM_WEIGHT();
    }
  }
    
  // allocate and initialize weights between layers.
  nn->weights = malloc(num_layers * sizeof(double**));
  nn->weights[0] = NULL; // no weights for the input layer.
  for(size_t i = 1; i < num_layers; ++i){
    size_t neurons_in_current = topology[i];
    size_t neurons_in_prev = topology[i-1];
    nn->weights[i] = malloc(neurons_in_current * sizeof(double*));
    for(size_t j = 0; j < neurons_in_current; ++j){
      nn->weights[i][j] = malloc(neurons_in_prev * sizeof(double));
      for(size_t k = 0; k < neurons_in_prev; ++k){
	nn->weights[i][j][k] = RANDOM_WEIGHT();
      }
    }
  }
  return nn;
}

// feed forward: calculates the output of the network given an input.
void
feedForward(NeuralNetwork *nn, double *input){
  // set the input layer.
  for(size_t i = 0; i < nn->topology[0]; ++i)
    nn->neurons[0][i] = input[i];
    
  // for every subsequent layer, compute the weighted sum and apply the sigmoid function.
  for(size_t l = 1; l < nn->num_layers; ++l){
    for(size_t j = 0; j < nn->topology[l]; ++j){
      double sum = nn->biases[l][j];
      for(size_t k = 0; k < nn->topology[l-1]; ++k){
	sum += nn->weights[l][j][k] * nn->neurons[l-1][k];
      }
      nn->neurons[l][j] = sigmoid(sum);
    }
  }
}

// utility function to print the output layer neurons.
void
printOutput(NeuralNetwork *nn){
  size_t output_layer = nn->num_layers - 1;
  for(size_t i = 0; i < nn->topology[output_layer]; ++i)
    printf("%f ", nn->neurons[output_layer][i]);
  printf("\n");
}

// the backpropagation routine calculates the error gradients (deltas) for each neuron,
// and then updates the weights and biases accordingly.
double
meanSquareErrorDerivative(double predicted, double target, size_t n){
  return (predicted - target) / n;
}

void
backPropagation(NeuralNetwork *nn, double *output, double learning_rate){
  size_t L = nn->num_layers;
    
  // Allocate an array of delta arrays (one per layer)
  double **deltas = malloc(L * sizeof(double*));
  for(size_t i = 0; i < L; ++i)
    deltas[i] = calloc(nn->topology[i], sizeof(double));
    
  // Calculate deltas for the output layer using our mean square error derivative function.
  size_t output_layer = L - 1;
  for(size_t i = 0; i < nn->topology[output_layer]; ++i){
    double out = nn->neurons[output_layer][i];
    // Compute the MSE derivative for this output neuron.
    double error_deriv = meanSquareErrorDerivative(out, output[i], nn->topology[output_layer]);
    deltas[output_layer][i] = error_deriv * sigmoidDerivative(out);
  }
    
  // Propagate deltas backwards for the hidden layers.
  for(size_t l = L - 2; l > 0; --l) {
    for(size_t i = 0; i < nn->topology[l]; ++i){
      double sum = 0.0;
      for(size_t j = 0; j < nn->topology[l+1]; ++j)
	sum += nn->weights[l+1][j][i] * deltas[l+1][j];
            
      double out = nn->neurons[l][i];
      deltas[l][i] = sum * sigmoidDerivative(out);
    }
  }
    
  // Update weights and biases using gradient descent.
  for(size_t l = 1; l < L; ++l){
    for(size_t i = 0; i < nn->topology[l]; ++i){
      for(size_t j = 0; j < nn->topology[l-1]; ++j)
	nn->weights[l][i][j] -= learning_rate * deltas[l][i] * nn->neurons[l-1][j];
            
      // Update bias.
      nn->biases[l][i] -= learning_rate * deltas[l][i];
    }
  }
    
  // Clean up delta arrays.
  for (size_t i = 0; i < L; ++i)
    free(deltas[i]);
  free(deltas);
}

// a simple training function that runs feedforward and backpropagation for a number of iterations.
void
train(NeuralNetwork *nn, double *input, double *output, double learning_rate){
  feedForward(nn, input);
  backPropagation(nn, output, learning_rate);
}

// free all dynamically allocated memory of the neural network.
void
freeNeuralNetwork(NeuralNetwork *nn){
  // free neuron arrays
  for(size_t i = 0; i < nn->num_layers; ++i)
    free(nn->neurons[i]);
  free(nn->neurons);
    
  // free biases
  for(size_t i = 1; i < nn->num_layers; ++i)
    free(nn->biases[i]);
  free(nn->biases);
    
  // free weights
  for(size_t i = 1; i < nn->num_layers; ++i){
    for(size_t j = 0; j < nn->topology[i]; ++j){
      free(nn->weights[i][j]);
    }
    free(nn->weights[i]);
  }
  free(nn->weights);
    
  free(nn->topology);
  free(nn);
}

void
printNetwork(NeuralNetwork *nn){
  // print architecture
  printf("\033[36mArchitecture\n\033[0m");
  printf("number of layers: %zu\n", nn->num_layers);
  printf("topology        : ");
  for(size_t i = 0; i < nn->num_layers; ++i){
    printf("%zu", nn->topology[i]);
    if(i < nn->num_layers-1)
      printf(", ");
  }
  printf("\n\n");

  // print neurons
  printf("\033[36mNeurons:\033[0m\n");
  printf("{ \n");
  for(size_t i = 0; i < nn->num_layers; ++i){
    printf("\t{ ");
    for(size_t j = 0; j < nn->topology[i]; ++j){
      printf("%f ", nn->neurons[i][j]);
      if(j < nn->topology[i]-1)
	printf("\t");
    }
    printf("}\n");
  }
  printf("}\n\n");

  // print biases
  printf("\033[36mBiases:\033[0m\n");
  printf("{ \n");
  for(size_t i = 1; i < nn->num_layers; ++i){
    printf("\t{ ");
    for(size_t j = 0; j < nn->topology[i]; ++j){
      printf("%f ", nn->biases[i][j]);
      if(j < nn->topology[i]-1)
	printf("\t");
    }
    printf("}\n");
  }
  printf("}\n\n");

  // print weights
  printf("\033[36mWeights:\033[0m\n");
  printf("{ \n");
  for(size_t i = 1; i < nn->num_layers; ++i){
    size_t neurons_in_current = nn->topology[i];
    size_t neurons_in_prev = nn->topology[i-1];
    printf("\t{ \n");
    for(size_t j = 0; j < neurons_in_current; ++j){
      printf("\t\t{ ");
      for(size_t k = 0; k < neurons_in_prev; ++k){
	printf("%f ", nn->weights[i][j][k]);
	if(k < neurons_in_prev-1)
	  printf("\t");
      }
      printf("}\n");
    }
    printf("\t}\n");
  }
  printf("}\n\n");
}

void
printTraining(NeuralNetwork *nn, size_t epoch){
  printf("\033[H");
  printf("\033[33mEpochs\033[0m          : %zu\n", epoch);
  printNetwork(nn);
}

// TODO
void
saveNetworkToFile(NeuralNetwork *nn, const char* filename){
  (void) nn;
  (void) filename;
}  

void
loadNetworkFromFile(NeuralNetwork *nn, const char* filename){
  (void) nn;
  (void) filename;
}

#endif
