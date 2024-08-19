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
  size_t  topology_size;
  size_t  weights_size;
  size_t  bias_size;
  size_t* topology;
  float*  weights;
  float*  bias;
} NeuralNetwork;

#define BUFFERSIZE 512
#define GET_CONTAINER_SIZE(container) sizeof((container))/sizeof((container)[0])

void load_training_file(Trainingset* set, const char* filename);
void print_trainingset(Trainingset set);
void free_trainingset(Trainingset set);
void initialize_model(NeuralNetwork* model, size_t* topology, size_t topology_size);
void print_model(NeuralNetwork model);
void free_model(NeuralNetwork model);
float randomize(float range_min, float range_max);

#endif

#ifdef ANNI_IMPLEMENTATION

void load_training_file(Trainingset* set, const char* filename){
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
    sscanf(buffer, "input %zu\n",   &set->input_size);
    sscanf(buffer, "output %zu\n",  &set->output_size);
    sscanf(buffer, "samples %zu\n", &set->samples);

    // save data given by file into the trainingset struct
    if(data_is_active && data_counter < set->samples){
      token = strtok(buffer, ",");
      set->data[data_counter * set->stride] = (float)atof(token);
      for(size_t i = 1; i < set->stride; ++i){
        	set->data[data_counter * set->stride + i] = (float)atof(strtok(NULL, ","));
      }
      data_counter+=1;
    }

    // toggle data flag and calculate rest of control values
    if(strcmp(buffer, "data\n") == 0){
      data_is_active     = true;
      set->stride         = set->input_size + set->output_size;
      set->container_size = (set->input_size + set->output_size) * set->samples;
      set->data           = malloc(sizeof(float) * set->container_size);
      token              = strtok(buffer, ",");
    }
  }
  fclose(file);
}

void print_trainingset(Trainingset set){
  printf("Trainingset:\n");
  printf("---------------\n");
  for(size_t i = 0; i < set.container_size; ++i){
    if(i % set.stride == 0 && i != 0) printf("\n");
    printf("%f ", set.data[i]);
  }
  printf("\n---------------\n");
}

void free_trainingset(Trainingset set){
  free(set.data);
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
}

void free_model(NeuralNetwork model){
  free(model.topology);
  free(model.weights);
  free(model.bias);
}

float randomize(float range_min, float range_max){
  return (float)rand() / (float)RAND_MAX * (range_max - range_min) + range_min;
}

#endif
