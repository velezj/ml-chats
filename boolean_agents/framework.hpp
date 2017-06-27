/*
 * Copyright Javier Velez <velezj@alum.mit.edu> June 2017
 * All Rights Reserved
 */


#include <bitset>
#include <vector>
#include <random>
#include <limits>
#include <cstdint>
#include <array>
#include <iostream>
#include <string>
#include <sstream>

//=======================================================================

/**
 * A world (as a bitset) with a rolling dynamics model
 */
template< uint32_t N, uint32_t Dynamics_M, uint32_t Dynamics_X >
class RollingWorld {
public:

  static const size_t Dynamics_XmodN = Dynamics_X % N;

  /**
   * The time step
   */
  size_t time_step;

  /**
   * The data
   */
  std::bitset<N> data;

public:


  /**
   * Move the time foward by one step
   */
  void step_time_forward() {
    this->time_step += 1;
    if( this->time_step % Dynamics_M == 0 ) {
      this->apply_dynamics_model();
    }
  }

protected:

  /**
   * Applies a dynamics model.
   * we jsut shift the data
   */
  void apply_dynamics_model() {
    this->data = ( this->data << Dynamics_XmodN | this->data >> ( N - Dynamics_XmodN ) );
  }
};

//=======================================================================
//=======================================================================

/**
 * An Agent is just a plan and a memory
 */
template< uint32_t M, typename ActionSpace >
struct Agent {
public:

  /**
   * The memory
   */
  std::bitset<M> memory;

  /**
   * The plan 
   */
  std::vector<ActionSpace> plan;
  
public:

protected:
  
};

//=======================================================================
//=======================================================================

/**
 * An ActionSpace defines an internal typename ::ActionSpace
 * and implements the possible actions achieveable, ever, by an
 * agent.
 */
template< uint32_t N, uint32_t M, typename world_t >
class BooleanActionSpace {

  /**
   * Make sure that M and N are small enough that we can fit in
   * a single packed unsigned long as our action space :)
   */
  static_assert( (M + N)*(M+N)*4 < std::numeric_limits<uint64_t>::max(),
		 "Sie of Memory (M) and World (N) too large for a flat 64bit action representation");
  
public:

  /**
   * Boolean operators enumeration
   */
  enum struct BooleanOp { And, Or, Nand, Nor };

  /**
   * The inner typename for hte aactual action space used by agents
   */
  struct ActionSpace {
    uint64_t packed_source_target_op;
  };

public:

  /**
   * Return a string representation of an action
   */
  std::string human_readable_action( const ActionSpace& action ) {

    // modulus to full action space
    uint64_t pack = action.packed_source_target_op % ( (M+N) * (M+N) * 4 );

    // unpack source nad target indices
    size_t res0 = ( pack % ( (M + N) * 4 ) );
    size_t source_index = ( pack / ( (M+N) * 4 ) );
    size_t res1 = ( res0 % 4 );
    size_t target_index = ( res0 / 4 );
    BooleanOp op = static_cast<BooleanOp>( res1 );

    std::ostringstream oss;
    std::string a0, a1;
    if( source_index < M ) {
      a0 = "M";
    } else {
      a0 = "W";
      source_index -= M;
    }
    if( target_index < M ) {
      a1 = "M";
    } else {
      a1 = "W";
      target_index -= M;
    }
    std::string opstring;
    switch( op ) {
    case BooleanOp::And  :
      opstring = "and";
      break;
    case BooleanOp::Or   :
      opstring = "or";
      break;
    case BooleanOp::Nand :
      opstring = "nand";
      break;
    case BooleanOp::Nor  :
      opstring = "nor";
      break;      
    }
    oss << a0 << source_index << "<-" << opstring << "(" << a0 << source_index << "," << a1 << target_index << ")";
    return oss.str();
  }

  /**
   * Perform a given action
   */
  void perform_action( world_t& world,
		       Agent<M, ActionSpace>& agent,
		       const ActionSpace& action ) {

    // full action space using modulus
    uint64_t pack = action.packed_source_target_op % ( (M+N) * (M+N) * 4 );

    // unpack source nad target indices
    size_t res0 = ( pack % ( (M + N) * 4 ) );
    size_t source_index = ( pack / ( (M+N) * 4 ) );
    size_t res1 = ( res0 % 4 );
    size_t target_index = ( res0 / 4 );
    BooleanOp op = static_cast<BooleanOp>( res1 );

    // ok, perform the boolean operation
    auto source = ( source_index < M ) ? agent.memory[ source_index ] : world.data[ source_index - M ];
    auto target = ( target_index < M ) ? agent.memory[ target_index ] : world.data[ target_index - M ];
    switch( op ) {
    case BooleanOp::And  :
      source &= target;
      break;
    case BooleanOp::Or   :
      source |= target;
      break;
    case BooleanOp::Nand :
      source = ~( source & target );
      break;
    case BooleanOp::Nor  :
      source = ~( source | target );
      break;      
    }
  }

public:

protected:
};

//=======================================================================
//=======================================================================

/**
 * The XORShift+ random number generator.
 * Code taken from Wikipedia article on XORShift
 */
std::array<uint64_t,2> _xorshiftplus_state;

/**
 * Seed the generator
 */
void xorshift128plus_seed( const std::array<uint64_t,2>& seed ) {
  _xorshiftplus_state = seed;
}

/**
 * grab a random integer from the xorshift+ algorithm
 */
uint64_t xorshift128plus() {
  uint64_t x = _xorshiftplus_state[0];
  uint64_t const y = _xorshiftplus_state[1];
  _xorshiftplus_state[0] = y;
  x ^= x << 23; // a
  _xorshiftplus_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
  return _xorshiftplus_state[1] + y;
}

/**
 * Grab a random float between 0,1
 */
float xorshift128plus_01() {
  uint64_t r = xorshift128plus();
  return static_cast<float>( static_cast<double>( r ) / static_cast<double>( std::numeric_limits<uint64_t>::max() ) );
}

//=======================================================================

/**
 * Updates a plan for an agent given
 *     the agent
 *     the world
 *     the reward
 */
template< typename agent_t, typename world_t, uint32_t PregenN >
class RewardTemperedUpdatePlan {
public:

  /**
   * Constructor
   * Pregenerates random numbers
   */
  RewardTemperedUpdatePlan() {
    this->_pregen_random();
  }

  /**
   * update the agents plan based on the reward
   */
  void update_plan( agent_t& agent,
		    const world_t& world,
		    const float& reward ) {

    // mutation rate is directly proportional to reward
    float p = 1.0 - reward;
    if( p < 0 ) {
      p = 0.0;
    }
    if( p > 1.0 ) {
      p = 1.0;
    }

    //std::cout << "[" << this->m_random_index << "$" << reward << "/" << p << " ";

    // no mutations if we are at max rewward
    if( p == 0.0 ) {
      return;
    }

    // mutate each plan element inependently
    bool mutated_any = false;
    for( size_t i = 0; i < agent.plan.size(); ++i ) {
      if( this->pregen_random() <= p ) {
	int mag = static_cast<int>( this->pregen_random() * 3 - 1);
	agent.plan[i].packed_source_target_op += mag;
	mutated_any = true;
      }
    }
    if( mutated_any ) {
      //std::cout << "!";
    }

    // see if we grow the plan
    if( agent.plan.size() > 0 && this->pregen_random() <= p ) {
      agent.plan.push_back( agent.plan[ agent.plan.size() - 1 ] );
    }

    // see if we shrink the plan
    if( agent.plan.size() > 1 && this->pregen_random() <= p ) {
      agent.plan.pop_back();
    }
  };

public:

  /**
   * A bunch of pre-generated random (0,1) numbers
   */
  std::array< float, PregenN > m_random;

  /**
   * index into which pregen random number we are at
   */
  size_t m_random_index;

  /**
   * pregenerate random floats
   */
  void _pregen_random() {
    for( size_t i = 0; i < PregenN; ++i ) {
      this->m_random[ i ] = xorshift128plus_01();
    }
  }

  /**
   * returns a random float using our pregenerated table
   */
  float pregen_random() {
    this->m_random_index = ( this->m_random_index + 1 ) % PregenN;
    return this->m_random[ this->m_random_index ];
  }
  
};

//=======================================================================
//=======================================================================

/**
 * A reward function returns the immediate rewarrds for an agent
 * and a world.
 * We expect rewards to be (0,1) bounded
 */
template< typename world_t, typename agent_t >
class AllOnesRewardFunction {
public:

public:

  /**
   * Returns the reward for a world and agent
   */
  float immediate_reward( const world_t& world,
			  const agent_t& agent ) {
    return static_cast<float>( world.data.count() ) / static_cast<float>( world.data.size() );
  }

protected:
};

//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

/**
 * A system is a compbination of world, agent, plan/update, and reward
 * We instantiate a single world and a set of agents, and
 * allow time to move forward, feeding hte reward and updating plans and
 * having agents take action as time moves.
 */
template< typename world_t,
	  typename agent_t,
	  typename action_space_t,
	  typename update_plan_t,
	  typename reward_function_t,
	  size_t AgentNum>
class System {
public:

  /**
   * The world, single instance
   */
  world_t world;

  /**
   * The reward function
   */
  reward_function_t reward_function;

  /**
   * The agents
   */
  std::array< agent_t, AgentNum > agents;

  /**
   * The actions space
   */
  action_space_t action_space;

  /**
   * The plan updater
   */
  update_plan_t update_plan;
  

public:

  /**
   * Run a certain number of time steps, returning the
   * resulting rewards for each step
   */
  std::vector<std::array<float,AgentNum> > run( const size_t& steps ) {
    std::vector<std::array<float,AgentNum> > rewards;
    for( size_t i = 0; i < steps; ++i ) {
      auto r = this->step_time_forward();
      rewards.push_back( r );
    }
    return rewards;
  }

  /**
   * Run time forward one step
   */
  std::array<float,AgentNum> step_time_forward() {

    // calculate rewards for each agent
    std::array<float, AgentNum> rewards;
    for( size_t i = 0; i < AgentNum; ++i ) {
      rewards[ i ] = this->reward_function.immediate_reward(this->world,
							    this->agents[i]);
    }

    // update hte plan for each agent
    for( size_t i = 0; i < AgentNum; ++i ) {
      this->update_plan.update_plan( this->agents[i],
				     this->world,
				     rewards[i] );
    }

    // update the world
    this->world.step_time_forward();

    // return the rewards
    return rewards;
  }

protected:
};

//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

/**
 * A simple system instantiation
 */
System<
  RollingWorld< 10, 10, 1 >,
  Agent< 4,
	 typename BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > >::ActionSpace >,
  BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > >,
  RewardTemperedUpdatePlan<
    Agent< 4,
	   typename BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > >::ActionSpace >,
    RollingWorld< 10, 10, 1 >,
    100000 >,
  AllOnesRewardFunction<
    RollingWorld< 10, 10, 1 >,
    Agent< 4,
	   typename BooleanActionSpace< 10, 4, RollingWorld< 10, 10, 1 > >::ActionSpace > >,
  1 > g_system01;
  
//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================
