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

#define BUFFERSIZE 512
#define GET_CONTAINER_SIZE(container) sizeof((container))/sizeof((container)[0])

void load_training_file(Trainingset* ts, const char* filename);
void print_trainingset(Trainingset ts);

#endif

#ifdef ANNI_IMPLEMENTATION

void load_training_file(Trainingset* ts, const char* filename){
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
    // skip comments in file
    if(buffer[0] == '#') continue; 

    // save values of input, output and samples to trainingset struct
    sscanf(buffer, "input %zu\n",   &ts->input_size);
    sscanf(buffer, "output %zu\n",  &ts->output_size);
    sscanf(buffer, "samples %zu\n", &ts->samples);

    // save data given by file into the trainingset struct
    if(data_is_active && data_counter < ts->samples){
      token = strtok(buffer, ",");
      ts->data[data_counter * ts->stride] = (float)atof(token);
      for(size_t i = 1; i < ts->stride; ++i){
        	ts->data[data_counter * ts->stride + i] = (float)atof(strtok(NULL, ","));
      }
      data_counter+=1;
    }

    // toggle data flag and calculate rest of control values
    if(strcmp(buffer, "data\n") == 0){
      data_is_active     = true;
      ts->stride         = ts->input_size + ts->output_size;
      ts->container_size = (ts->input_size + ts->output_size) * ts->samples;
      ts->data           = malloc(sizeof(float) * ts->container_size);
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

#endif
