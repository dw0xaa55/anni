#ifndef _ANNI_H_
#define _ANNI_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef enum { false, true } bool;

typedef struct {
  size_t input_size;
  size_t output_size;
  size_t samples;
  size_t container_size;
  size_t stride;
  float* data;
} Trainingset;

typedef struct {
  size_t size;
  size_t stride;
  size_t samples;
  float* data;
} DataContainer;

typedef struct {
  size_t  topology_size;
  size_t  weights_size;
  size_t  bias_size;
  size_t  neurons_size;
  size_t* topology;
  float*  weights;
  float*  bias;
  float*  neurons;
} NeuralNetwork;

#define BUFFERSIZE 512
#define GET_CONTAINER_SIZE(container) sizeof((container))/sizeof((container)[0])

void  load_training_file(Trainingset* trainingset, const char* filename);
void  print_trainingset(Trainingset trainingset);
void  free_trainingset(Trainingset trainingset);
void  get_inputs_from_trainingset(DataContainer* container, Trainingset trainingset);
void  get_outputs_from_trainingset(DataContainer* container, Trainingset trainingset);
void  free_container(DataContainer container);
void  print_container(DataContainer container);
void  initialize_model(NeuralNetwork* model, size_t* topology, size_t topology_size);
void  print_model(NeuralNetwork model);
void  free_model(NeuralNetwork model);
float randomize(float range_min, float range_max);
float sigmoid(float x);
void  feed_forward(NeuralNetwork* model);
float mean_square_error(NeuralNetwork model, Trainingset trainingset);
float finite_difference(NeuralNetwork model, float epsilon, float learningrate);

#endif

#ifdef ANNI_IMPLEMENTATION

void load_training_file(Trainingset* trainingset, const char* filename){
  FILE*  file = fopen(filename, "r");
  char   buffer[BUFFERSIZE];
  bool   data_is_active = false;
  size_t data_counter   = 0;
  char*  token          = "";

  if(!file){
    printf("Unable to open file \"%s\".\n", filename);
    exit(1);
  }
  
  while(fgets(buffer, BUFFERSIZE, file) != NULL) {
    // skip commenset in file
    if(buffer[0] == '#') continue;

    // save values of input, output and samples to trainingset struct
    sscanf(buffer, "input %zu\n",   &trainingset->input_size);
    sscanf(buffer, "output %zu\n",  &trainingset->output_size);
    sscanf(buffer, "samples %zu\n", &trainingset->samples);

    // save data given by file into the trainingset struct
    if(data_is_active && data_counter < trainingset->samples){
      token = strtok(buffer, ",");
      trainingset->data[data_counter * trainingset->stride] = (float)atof(token);
      for(size_t i = 1; i < trainingset->stride; ++i){
	trainingset->data[data_counter * trainingset->stride + i] = (float)atof(strtok(NULL, ","));
      }
      data_counter+=1;
    }

    // toggle data flag and calculate rest of control values
    if(strcmp(buffer, "data\n") == 0){
      data_is_active              = true;
      trainingset->stride         = trainingset->input_size + trainingset->output_size;
      trainingset->container_size = trainingset->stride * trainingset->samples;
      trainingset->data           = malloc(sizeof(float) * trainingset->container_size);
    }
  }
  fclose(file);
}

void print_trainingset(Trainingset trainingset){
  printf("Trainingset:\n");
  printf("---------------\n");
  for(size_t i = 0; i < trainingset.container_size; ++i){
    if(i % trainingset.stride == 0 && i != 0) printf("\n");
    printf("%f ", trainingset.data[i]);
  }
  printf("\n---------------\n");
}

void free_trainingset(Trainingset trainingset){
  free(trainingset.data);
}

void get_inputs_from_trainingset(DataContainer* container, Trainingset trainingset){
  container->size    = trainingset.samples * trainingset.input_size;
  container->stride  = trainingset.input_size;
  container->samples = trainingset.samples;
  container->data    = malloc(sizeof(float) * container->size);
  for(size_t i = 0; i < trainingset.samples; ++i)
    for(size_t j = 0; j < trainingset.input_size; ++j)
      container->data[i*container->stride+j] = trainingset.data[i*trainingset.stride+j];
}

void get_outputs_from_trainingset(DataContainer* container, Trainingset trainingset){
  container->size    = trainingset.samples * trainingset.output_size;
  container->stride  = trainingset.output_size;
  container->samples = trainingset.samples;
  container->data    = malloc(sizeof(float) * container->size);  
  for(size_t i = 0; i < trainingset.samples; ++i)
    for(size_t j = trainingset.input_size; j < (trainingset.input_size+trainingset.output_size); ++j)
      container->data[i * container->stride + (j - trainingset.input_size)] = trainingset.data[i*trainingset.stride+j];
}

void print_container(DataContainer container){
  printf("Container:\n");
  for(size_t i = 0; i < container.samples; ++i){
    for(size_t j = 0; j < container.stride; ++j){
      printf("%f ", container.data[i*container.stride+j]);
    }
    printf("\n");
  }
}

void free_container(DataContainer container){
  free(container.data);
}

void initialize_model(NeuralNetwork* model, size_t* topology, size_t topology_size){
  model->topology_size = topology_size;

  model->topology = malloc(sizeof(size_t) * model->topology_size);
  for(size_t i = 0; i < model->topology_size; ++i)
    model->topology[i] = topology[i];
  
  model->weights_size = 0;
  for(size_t i = 0; i < model->topology_size - 1; ++i)
    model->weights_size += model->topology[i] * model->topology[i+1];
  model->weights = malloc(sizeof(float) * model->weights_size);
  for(size_t i = 0; i < model->weights_size; ++i)
    model->weights[i] = randomize(0.0f, 1.0f);

  model->bias_size = model->topology_size - 1;
  model->bias = malloc(sizeof(float) * model->bias_size);
  for(size_t i = 0; i < model->bias_size; ++i)
    model->bias[i] = randomize(0.0f, 1.0f);

  model->neurons_size = 0;
  for(size_t i = 0; i < model->topology_size; ++i)
    model->neurons_size += model->topology[i];
  model->neurons = malloc(sizeof(float) * model->neurons_size);
  for(size_t i = 0; i < model->neurons_size; ++i)
    model->neurons[i] = 0.0f;
}

void print_model(NeuralNetwork model){
  printf("# model\n");
  printf("topology_size %zu\n", model.topology_size);
  printf("weights_size %zu\n",  model.weights_size);
  printf("bias_size %zu\n",     model.bias_size);
  printf("topology ");
  for(size_t i = 0; i < model.topology_size; ++i)
    printf("%zu ", model.topology[i]);
  printf("\n");
  printf("weights ");
  for(size_t i = 0; i < model.weights_size; ++i)
    printf("%f ",  model.weights[i]);
  printf("\n");
  printf("bias ");
  for(size_t i = 0; i < model.bias_size; ++i)
    printf("%f ",  model.bias[i]);
  printf("\n");
  printf("neurons ");
  for(size_t i = 0; i < model.neurons_size; ++i)
    printf("%f ",  model.neurons[i]);
  printf("\n");
}

void free_model(NeuralNetwork model){
  free(model.topology);
  free(model.weights);
  free(model.bias);
  free(model.neurons);
}

float randomize(float range_min, float range_max){
  return (float)rand() / (float)RAND_MAX * (range_max - range_min) + range_min;
}

float sigmoid(float x){
  return 1.0f/(1.0f + expf(-x));
}

void feed_forward(NeuralNetwork* model){
  size_t neuron_stride = 0;
  size_t weight_stride = 0;
  size_t layer_stride  = model->topology[0];
  // creates two work matrices each step iterating through topology
  for(size_t i = 0; i < model->topology_size - 1; ++i){
    float work_matrix_1[model->topology[i]];
    float work_matrix_2[model->topology[i] * model->topology[i+1]];

    // populate matrices
    for(size_t j = neuron_stride; j < neuron_stride + model->topology[i]; ++j)
      work_matrix_1[j - neuron_stride] = model->neurons[j];
    for(size_t j = weight_stride; j < weight_stride + model->topology[i] * model->topology[i+1]; ++j)
      work_matrix_2[j-weight_stride] = model->weights[j];
    
    // feed forward
    size_t next_row = 0;
    for(size_t j = 0; j < model->topology[i + 1]; ++j){
      for(size_t k = 0; k < model->topology[i]; ++k)
	model->neurons[layer_stride + j] += work_matrix_1[k] * work_matrix_2[next_row + k]; // multiply matrices
      model->neurons[layer_stride + j] += model->bias[i];                                   // add bias to neuron
      model->neurons[layer_stride + j] = sigmoid(model->neurons[layer_stride + j]);         // modify neuron with sigmoid activation function
      next_row += model->topology[i];
    }

    // stride increments
    neuron_stride += model->topology[i];
    weight_stride += model->topology[i] * model->topology[i + 1];
    layer_stride  += model->topology[i+1];
  }
}

float mean_square_error(NeuralNetwork model, Trainingset trainingset){
  (void) model;
  (void) trainingset;
  return 0.0f;
}

float finite_difference(NeuralNetwork model, float epsilon, float learningrate){
  (void) model;
  (void) epsilon;
  (void) learningrate;
  return 0.0f;
}

#endif
