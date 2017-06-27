/*
 * Copyright Javier Velez <velezj@alum.mit.edu> June 2017
 * All Rights Reserved
 */


#include "framework.hpp"
#include <iostream>
#include <ctime>

int main(void) {

  size_t NUM_STEPS = 10000000;

  // start the timer
  // struct timespec start, start_run, finish;
  // clock_gettime(CLOCK_REALTIME, &start);
  const clock_t start = clock();

  // seed the random number generator
  xorshift128plus_seed( { clock() + 100010290875ul, clock() + 95 } );

  // initialize hte world to a random set of 0,1
  for( size_t i = 0; i < g_system01.world.data.size(); ++i ) {
    bool b = ( xorshift128plus_01() < 0.5 );
    g_system01.world.data[ i ] = b;
    std::cout << static_cast<int>(b);
  }
  std::cout << std::endl;

  // initialize the agent's plan to a random set of actions
  const uint64_t max_action = (10+4)*(10+4)*2;
  const size_t max_init_plan_size = 10;
  size_t init_plan_size = 1 + static_cast<size_t>(xorshift128plus_01() * (max_init_plan_size - 1) );
  std::cout << "Initial Plan: ";
  for( size_t i = 0; i < init_plan_size; ++i ) {
    BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > > as;
    BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > >::ActionSpace a;
    a.packed_source_target_op = static_cast<uint64_t>( xorshift128plus_01() * max_action );
    g_system01.agents[0].plan.push_back( a );
    std::cout << a.packed_source_target_op << ":" << as.human_readable_action(a) << "  ";
  }
  std::cout << std::endl;

  // pregenerate random floats for updates
  g_system01.update_plan._pregen_random();
  // std::cout << "Pregen Randoms: ";
  // for( auto &&r : g_system01.update_plan.m_random ) {
  //   std::cout << r << " ";
  // }
  // std::cout << std::endl;

  //clock_gettime(CLOCK_REALTIME, &start_run);
  const clock_t start_run = clock();
  std::vector< std::array< float, 1> > rewards;
  size_t steps_per_run = 10000;
  if( steps_per_run > NUM_STEPS ) {
    steps_per_run = NUM_STEPS;
  }
  size_t num_runs = NUM_STEPS / steps_per_run;
  for( size_t i = 0; i < num_runs; ++i ) {
    rewards = g_system01.run( steps_per_run );
  }
  // for( auto &&rs : rewards ) {
  //   std::cout << rs[0] << std::endl;
  // }

  // compute time to finish
  // clock_gettime(CLOCK_REALTIME, &finish);
  // double elapsed = (finish.tv_sec - start.tv_sec);
  // elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  // double elapsed_run = (finish.tv_sec - start_run.tv_sec);
  // elapsed_run += (finish.tv_nsec - start_run.tv_nsec) / 1000000000.0;
  const clock_t finish = clock();
  double elapsed = (double)(finish - start) / CLOCKS_PER_SEC;

  // compute runs per second
  double runs_per_sec = NUM_STEPS / elapsed;
  
  std::cout << "Total time: " << elapsed << " (" << runs_per_sec << ") steps/sec" << std::endl;

  // print out hte last 10 rewards
  std::cout << "First 10 Rewards: ";
  for( size_t i = 0; i < 10; ++i ) {
    std::cout << rewards[ i ][0] << " , ";
  }
  std::cout << std::endl;
  std::cout << "Last 10 Rewards: ";
  for( size_t i = 0; i < 10; ++i ) {
    std::cout << rewards[ rewards.size() - i - 1 ][0] << " , ";
  }
  std::cout << std::endl;

  // print last/current plan
  std::cout << "Last Plan: ";
  for( auto &&a : g_system01.agents[0].plan ) {
    BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > > as;
    std::cout << a.packed_source_target_op << ":" << as.human_readable_action(a) << "  ";
  }
  std::cout << std::endl;

  // print last memory
  std::cout << "Last Memory: " << g_system01.agents[0].memory << std::endl;

  // print last world
  std::cout << "Last World: " << g_system01.world.data << std::endl;
}
