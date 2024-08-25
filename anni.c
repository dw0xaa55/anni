/* DESCRIPTION: Artificial Neural Network Intelligence   */
/* AUTHOR     : C. Huffenbach                            */
/* VERSION    : 0.3                                      */
/* DATE       : 08/2024                                  */

/* TODO:                                                 */
/*   - [X] think of trainingset architecture             */
/*   - [X] create Network struct                         */
/*   - [X] load trainingsets from file                   */
/*   - [X] option to print networks                      */
/*   - [X] implement feed_forward                        */
/*   - [ ] implement mean_square_error                   */
/*   - [ ] implement finite_difference                   */
/*   - [ ] save trained network to file                  */
/*   - [ ] option to load trained networks from file     */


#define ANNI_IMPLEMENTATION
#include "anni.h"

int main(int argc, char** argv){
  (void) argc;
  (void) argv;

  // setup
  Trainingset ts;
  NeuralNetwork nn;
  DataContainer inputs;
  DataContainer outputs;
  size_t topology[] = { 2,2,1 };

  load_training_file(&ts,"trainingfile.ts");
  print_trainingset(ts);
  initialize_model(&nn, topology, GET_CONTAINER_SIZE(topology));

  // feed forward
  get_inputs_from_trainingset(&inputs, ts);
  get_outputs_from_trainingset(&outputs, ts);
  print_container(inputs);
  /* print_container(outputs); */
  printf("-------------------------\n");
  printf("feed forward debug output:\n");
  nn.neurons[0] = inputs.data[2];
  nn.neurons[1] = inputs.data[3];
  feed_forward(&nn);
  print_model(nn);

  // clean up
  free_container(inputs);
  free_container(outputs);
  free_model(nn);
  free_trainingset(ts);
  
  return 0;
}
