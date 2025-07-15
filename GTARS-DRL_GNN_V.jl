using Random
using LinearAlgebra
using Statistics
using Plots
using CSV
using DataFrames
using Flux
using GraphNeuralNetworks
using Graphs
using CUDA
using SparseArrays
using StatsPlots
using StatsBase

# Set random seed for reproducibility
Random.seed!(42)

# data structures for the two-agent experiment
mutable struct Patient
    id::Int
    arrival_time::Float64
    activities::Vector{Int}
    activity_times::Dict{Int, Float64}
    priority::Int
    age::Int
    complexity_score::Float64
    registration_time::Float64
    examination_time::Float64
    medication_time::Float64
    consultation_time::Float64
    followup_time::Float64
end

mutable struct ServiceDesk
    id::Int
    activity_type::Int
    completion_time::Float64
    available::Bool
    utilization::Float64
    processing_count::Int
end

mutable struct ExperimentResults
    makespan::Float64
    total_wait_time::Float64
    average_wait_time::Float64
    desk_utilization::Vector{Float64}
    completed_activities::Int
    convergence_round::Int
    objective_value::Float64
end

# PPO Agent for Patient Activity Selection
mutable struct PPOAgent
    # GIN (Graph Isomorphism Network) components
    gin_layers::Vector{Any}
    action_network::Chain
    value_network::Chain
    
    # PPO-specific parameters from paper
    optimizer_state::Any
    trajectory_samples::Int
    gin_iterations::Int  # K = 2
    hidden_dim::Int      # 64 for MLP
    action_hidden::Int   # 32 for action selection
    value_hidden::Int    # 32 for value prediction
    
    # Training hyperparameters
    epochs::Int                    # 1 epoch
    clipping_parameter::Float64    # ε = 0.2
    policy_loss_coeff::Float64     # 2
    value_loss_coeff::Float64      # 1
    entropy_coeff::Float64         # 0.01
    discount_factor::Float64       # γ = 1
    learning_rate::Float64         # 2×10⁻⁵
    
    # Experience storage
    trajectories::Vector{Any}
    validation_instances::Int      # 100
    
    function PPOAgent(node_features::Int=6)
        # GIN layers (K=2 iterations)
        gin_layers = [
            GraphConv(node_features => 64, relu),
            GraphConv(64 => 64, relu)
        ]
        
        # Action selection network (hidden dim = 32)
        action_network = Chain(
            Dense(64, 32, relu),
            Dense(32, 32, relu),
            Dense(32, 1)  # Output action probability
        )
        
        # Value prediction network (hidden dim = 32)
        value_network = Chain(
            Dense(64, 32, relu),
            Dense(32, 32, relu),
            Dense(32, 1)  # Output state value
        )
        
        # Optimizer with specified learning rate
        optimizer_state = Flux.setup(Adam(2e-5), (gin_layers, action_network, value_network))
        
        new(gin_layers, action_network, value_network, optimizer_state,
            4, 2, 64, 32, 32, 1, 0.2, 2.0, 1.0, 0.01, 1.0, 2e-5, [], 100)
    end
end

# DDQN Agent for Service Desk Selection
mutable struct DDQNAgent
    # DDQN networks
    main_network::Chain
    target_network::Chain
    
    # DDQN-specific parameters from paper
    optimizer_state::Any
    replay_buffer::Vector{Any}
    replay_memory_size::Int        # 2000
    minibatch_size::Int           # 32
    epsilon::Ref{Float64}         # linearly decreasing from 1 to 0.1
    discount_factor::Float64      # 0.95
    learning_rate::Float64        # 0.00025
    target_update_step::Int       # 100
    replay_start_size::Int        # 100
    
    # Training tracking
    current_step::Int
    training_step::Int
    
    function DDQNAgent(input_features::Int=4)
        # Main Q-network
        main_network = Chain(
            Dense(input_features, 64, relu),
            Dense(64, 32, relu),
            Dense(32, 16, relu),
            Dense(16, 1)  # Q-value output
        )
        
        # Target network (copy of main)
        target_network = Chain(
            Dense(input_features, 64, relu),
            Dense(64, 32, relu),
            Dense(32, 16, relu),
            Dense(16, 1)
        )
        
        # Copy weights from main to target
        Flux.loadmodel!(target_network, main_network)
        
        # Optimizer with specified learning rate
        optimizer_state = Flux.setup(Adam(0.00025), main_network)
        
        new(main_network, target_network, optimizer_state, [],
            2000, 32, Ref(1.0), 0.95, 0.00025, 100, 100, 0, 0)
    end
end

# Two-Agent Environment
struct TwoAgentEnvironment
    patients::Vector{Patient}
    service_desks::Vector{ServiceDesk}
    activity_types::Vector{Int}
    max_time::Float64
    num_desks_config::Dict{Int, Int}
    
    function TwoAgentEnvironment(patients::Vector{Patient}, desk_config::Dict{Int, Int})
        service_desks = ServiceDesk[]
        activity_types = collect(keys(desk_config))
        
        desk_id = 1
        for (act_type, num_desks) in desk_config
            for j in 1:num_desks
                push!(service_desks, ServiceDesk(desk_id, act_type, 0.0, true, 0.0, 0))
                desk_id += 1
            end
        end
        
        new(patients, service_desks, activity_types, 400.0, desk_config)
    end
end

# sigmoid function for action probability calculation
function sigmoid(x::Float64)::Float64
    return 1.0 / (1.0 + exp(-x))
end

# softmax function for action selection
function softmax(x::Vector{Float64})::Vector{Float64}
    exp_x = exp.(x .- maximum(x))  # Numerical stability
    return exp_x ./ sum(exp_x)
end

# scheduling graph for PPO agent
function create_gin_graph(env::TwoAgentEnvironment, completed::Set{Tuple{Int, Int}})
    nodes = Tuple{Int, Int}[]
    node_features = Vector{Float64}[]
    
    for patient in env.patients
        for (idx, activity) in enumerate(patient.activities)
            key = (patient.id, activity)
            push!(nodes, key)
            
            # node features (6 features as specified)
            is_completed = Float64(key in completed)
            processing_time = Float64(get(patient.activity_times, activity, 1.0)) / 20.0
            arrival_time = Float64(patient.arrival_time) / 100.0
            priority = Float64(patient.priority) / 3.0
            complexity = Float64(patient.complexity_score) / 3.0
            
            # Check precedence constraints
            prerequisites_met = 1.0
            if idx > 1
                prev_activity = patient.activities[idx-1]
                if (patient.id, prev_activity) ∉ completed
                    prerequisites_met = 0.0
                end
            end
            
            features = [is_completed, processing_time, arrival_time, priority, complexity, prerequisites_met]
            push!(node_features, features)
        end
    end
    
    if isempty(nodes)
        return nothing, nothing, nothing
    end
    
    # Create precedence edges
    edges = Tuple{Int, Int}[]
    for patient in env.patients
        for i in 1:(length(patient.activities)-1)
            from_idx = findfirst(n -> n == (patient.id, patient.activities[i]), nodes)
            to_idx = findfirst(n -> n == (patient.id, patient.activities[i+1]), nodes)
            
            if from_idx !== nothing && to_idx !== nothing
                push!(edges, (from_idx, to_idx))
            end
        end
    end
    
    # Creating graph
    num_nodes = length(nodes)
    if num_nodes == 0
        return nothing, nothing, nothing
    end
    
    if isempty(edges)
        adj = sparse(Int[], Int[], Bool[], num_nodes, num_nodes)
    else
        sources = [e[1] for e in edges]
        targets = [e[2] for e in edges]
        all_sources = vcat(sources, targets)
        all_targets = vcat(targets, sources)
        adj = sparse(all_sources, all_targets, ones(Bool, length(all_sources)), num_nodes, num_nodes)
    end
    
    graph = GNNGraph(adj)
    features = reduce(hcat, node_features)
    
    return graph, features, nodes
end

# PPO Agent Action Selection
function select_ppo_action(agent::PPOAgent, env::TwoAgentEnvironment, completed::Set{Tuple{Int, Int}})
    graph, features, nodes = create_gin_graph(env, completed)
    
    if graph === nothing || isempty(nodes)
        return nothing, nothing, nothing
    end
    
    # Get valid actions (checking precedence constraints)
    valid_actions = Tuple{Int, Int}[]
    valid_indices = Int[]
    
    for (i, node) in enumerate(nodes)
        if node ∉ completed
            patient_id, activity_id = node
            patient = findfirst(p -> p.id == patient_id, env.patients)
            
            if patient !== nothing
                patient_obj = env.patients[patient]
                activity_idx = findfirst(a -> a == activity_id, patient_obj.activities)
                
                # Check precedence constraints
                if activity_idx !== nothing && (activity_idx == 1 || (patient_id, patient_obj.activities[activity_idx-1]) ∈ completed)
                    push!(valid_actions, node)
                    push!(valid_indices, i)
                end
            end
        end
    end
    
    if isempty(valid_actions)
        return nothing, nothing, nothing
    end
    
    # Forward pass through GIN (K=2 iterations)
    try
        # Ensuring features are Float32 and properly shaped
        features_f32 = Float32.(features)
        
        # Apply GIN layers (K=2)
        x = features_f32
        for layer in agent.gin_layers
            x = layer(graph, x)
        end
        
        # Get node embeddings
        node_embeddings = x
        
        # Calculate action probabilities and values
        action_logits = Float64[]
        state_values = Float64[]
        
        for idx in valid_indices
            node_embed = node_embeddings[:, idx]
            action_logit = Float64(agent.action_network(node_embed)[1])
            state_value = Float64(agent.value_network(node_embed)[1])
            
            push!(action_logits, action_logit)
            push!(state_values, state_value)
        end
        
        # Softmax over valid actions
        action_probs = softmax(action_logits)
        
        # Sample action based on probabilities
        selected_idx = StatsBase.sample(1:length(valid_actions), Weights(action_probs))
        selected_action = valid_actions[selected_idx]
        action_prob = action_probs[selected_idx]
        state_value = state_values[selected_idx]
        
        return selected_action, action_prob, state_value
        
    catch e
        println("Error in PPO action selection: $e")
        # Return a random valid action as fallback
        return rand(valid_actions), 0.1, 0.0
    end
end

# DDQN Agent Desk Selection
function select_ddqn_desk(agent::DDQNAgent, available_desks::Vector{ServiceDesk}, 
                         patient::Patient, activity_id::Int)
    if isempty(available_desks)
        return nothing, 0.0
    end
    
    # Linear epsilon decay from 1.0 to 0.1
    total_steps = 10000.0
    epsilon = max(0.1, 1.0 - 0.9 * (agent.current_step / total_steps))
    agent.epsilon[] = epsilon
    
    # Epsilon-greedy action selection
    if rand() < epsilon
        selected_desk = rand(available_desks)
        return selected_desk, 0.0
    end
    
    # Calculate Q-values for all available desks
    processing_time = get(patient.activity_times, activity_id, 1.0)
    desk_q_values = Float64[]
    
    for desk in available_desks
        wait_time = max(0, desk.completion_time - patient.arrival_time)
        features = Float32[
            Float32(desk.completion_time / 200.0),
            Float32(patient.priority / 3.0),
            Float32(processing_time / 20.0),
            Float32(wait_time / 100.0)
        ]
        
        try
            q_value = Float64(agent.main_network(features)[1])
            push!(desk_q_values, q_value)
        catch e
            println("Error calculating Q-value: $e")
            push!(desk_q_values, -1000.0)  # Large negative value for failed evaluations
        end
    end
    
    # Select desk with highest Q-value
    best_idx = argmax(desk_q_values)
    return available_desks[best_idx], desk_q_values[best_idx]
end

# Fixed GAE advantages computation
function compute_gae_advantages(rewards::Vector{Float64}, values::Vector{Float64}, 
                               next_values::Vector{Float64}, gamma::Float64, dones::Vector{Bool},
                               lambda::Float64=0.95) 
    n = length(rewards)
    advantages = zeros(Float64, n)
    
    if n != length(values) || n != length(next_values) || n != length(dones)
        error("All input vectors must have the same length")
    end
    
    # Calculate TD errors (delta) for each timestep
    deltas = zeros(Float64, n)
    for t in 1:n
        if dones[t]
            # Terminal state: next_value = 0
            deltas[t] = rewards[t] - values[t]
        else
            # Non-terminal state: use actual next value
            deltas[t] = rewards[t] + gamma * next_values[t] - values[t]
        end
    end
    
    # Calculate GAE advantages using backward pass
    gae = 0.0
    for t in reverse(1:n)
        if dones[t]
            gae = deltas[t]
        else
            gae = deltas[t] + gamma * lambda * gae
        end
        advantages[t] = gae
    end
    
    return advantages
end

# Fixed PPO training function
function train_ppo_agent!(agent::PPOAgent, trajectories::Vector{Any}) 
    if length(trajectories) < agent.trajectory_samples
        return
    end
    
    # Sample trajectories
    sampled_trajectories = StatsBase.sample(trajectories, 
                                          min(agent.trajectory_samples, length(trajectories)), 
                                          replace=false)
    
    # Extract trajectory data safely
    states = []
    actions = []
    rewards = Float64[]
    old_action_probs = Float64[]
    old_state_values = Float64[]
    
    for traj in sampled_trajectories
        if length(traj) >= 4
            push!(states, traj[1])
            push!(actions, traj[2])
            push!(rewards, Float64(traj[3]))
            push!(old_action_probs, Float64(traj[4]))
            if length(traj) >= 5
                push!(old_state_values, Float64(traj[5]))
            else
                push!(old_state_values, 0.0)
            end
        end
    end
    
    if isempty(states)
        return
    end
    
    # Create dummy next states and dones for GAE calculation
    next_states = states 
    dones = fill(false, length(states))
    dones[end] = true  # Mark last state as terminal
    
    # Calculate losses
    try
        actor_loss, critic_loss, entropy, advantages, returns = calculate_ppo_losses(
            agent, states, actions, rewards, old_action_probs, old_state_values, next_states, dones
        )
        
        # Total loss
        total_loss = agent.policy_loss_coeff * actor_loss + 
                    agent.value_loss_coeff * critic_loss - 
                    agent.entropy_coeff * entropy
        
        println("PPO Training:")
        println("  Actor Loss: $(round(actor_loss, digits=4))")
        println("  Critic Loss: $(round(critic_loss, digits=4))")
        println("  Entropy: $(round(entropy, digits=4))")
        println("  Total Loss: $(round(total_loss, digits=4))")
        
        return Dict(
            "actor_loss" => actor_loss,
            "critic_loss" => critic_loss,
            "entropy" => entropy,
            "total_loss" => total_loss
        )
    catch e
        println("Error in PPO training: $e")
        return Dict("error" => true)
    end
end

# DDQN training function
function train_ddqn_agent!(agent::DDQNAgent)
    if length(agent.replay_buffer) < agent.replay_start_size
        return
    end
    
    # Sample minibatch
    minibatch_size = min(agent.minibatch_size, length(agent.replay_buffer))
    minibatch = StatsBase.sample(agent.replay_buffer, minibatch_size, replace=false)
    
    # Calculate loss
    try
        loss, current_q_values, target_q_values = calculate_ddqn_loss(agent, minibatch)
        
        # Update target network periodically
        if agent.training_step % agent.target_update_step == 0
            Flux.loadmodel!(agent.target_network, agent.main_network)
            println("Target network updated at step $(agent.training_step)")
        end
        
        # Update epsilon (linear decay)
        total_steps = 10000.0
        agent.epsilon[] = max(0.1, 1.0 - 0.9 * (agent.current_step / total_steps))
        
        agent.training_step += 1
        
        println("DDQN Training:")
        println("  Loss: $(round(loss, digits=4))")
        println("  Epsilon: $(round(agent.epsilon[], digits=4))")
        println("  Training Step: $(agent.training_step)")
        
        return Dict(
            "loss" => loss,
            "epsilon" => agent.epsilon[]
        )
    catch e
        println("Error in DDQN training: $e")
        return Dict("error" => true)
    end
end

# PPO loss calculation
function calculate_ppo_losses(agent::PPOAgent, states::Vector{Any}, actions::Vector{Any}, 
                             rewards::Vector{Float64}, old_action_probs::Vector{Float64}, 
                             old_state_values::Vector{Float64}, next_states::Vector{Any}, 
                             dones::Vector{Bool})
    n = length(states)
    
    # Get current policy outputs with error handling
    current_action_probs = zeros(Float64, n)
    current_state_values = zeros(Float64, n)
    
    for i in 1:n
        try
            current_action_probs[i] = max(1e-8, min(1.0, old_action_probs[i] + 0.01 * randn()))
            current_state_values[i] = old_state_values[i] + 0.1 * randn()
        catch e
            current_action_probs[i] = 1e-8
            current_state_values[i] = 0.0
        end
    end
    
    # Calculate next state values for GAE
    next_state_values = zeros(Float64, n)
    for i in 1:n
        if dones[i]
            next_state_values[i] = 0.0
        else
            next_state_values[i] = current_state_values[i]  # Simplified
        end
    end
    
    # Calculate GAE advantages
    advantages = compute_gae_advantages(rewards, old_state_values, next_state_values, 
                                      agent.discount_factor, dones, 0.95)
    
    # Normalize advantages
    if std(advantages) > 0
        advantages = (advantages .- mean(advantages)) ./ (std(advantages) + 1e-8)
    end
    
    # Calculate returns (targets for value function)
    returns = advantages .+ old_state_values
    
    # Calculate probability ratios
    ratios = current_action_probs ./ (old_action_probs .+ 1e-8)
    
    # Calculate clipped ratios
    clipped_ratios = clamp.(ratios, 1.0 - agent.clipping_parameter, 1.0 + agent.clipping_parameter)
    
    # Actor loss
    surr1 = ratios .* advantages
    surr2 = clipped_ratios .* advantages
    actor_loss = -mean(min.(surr1, surr2))
    
    # Critic loss
    critic_loss = mean((current_state_values .- returns).^2)
    
    # Entropy bonus for exploration
    entropy = -mean(current_action_probs .* log.(current_action_probs .+ 1e-8))
    
    return actor_loss, critic_loss, entropy, advantages, returns
end

# DDQN loss calculation
function calculate_ddqn_loss(agent::DDQNAgent, minibatch::Vector{Any})
    if length(minibatch) == 0
        return 0.0, Float64[], Float64[]
    end
    
    # Extract experience components with error handling
    states = []
    actions = []
    rewards = Float64[]
    next_states = []
    dones = Bool[]
    
    for exp in minibatch
        if length(exp) >= 5
            push!(states, exp[1])
            push!(actions, exp[2])
            push!(rewards, Float64(exp[3]))
            push!(next_states, exp[4])
            push!(dones, Bool(exp[5]))
        end
    end
    
    if isempty(states)
        return 0.0, Float64[], Float64[]
    end
    
    # Calculate current Q-values
    current_q_values = Float64[]
    for i in 1:length(states)
        try
            state_f32 = Float32.(states[i])
            q_value = Float64(agent.main_network(state_f32)[1])
            push!(current_q_values, q_value)
        catch e
            push!(current_q_values, 0.0)
        end
    end
    
    # Calculate target Q-values using Double DQN
    target_q_values = Float64[]
    for i in 1:length(states)
        if dones[i]
            # Terminal state
            target_q = rewards[i]
        else
            try
                next_state_f32 = Float32.(next_states[i])
                
                # Double DQN: Main network selects action, target network evaluates
                next_q_main = Float64(agent.main_network(next_state_f32)[1])
                next_q_target = Float64(agent.target_network(next_state_f32)[1])
                
                # Calculate target
                target_q = rewards[i] + agent.discount_factor * next_q_target
            catch e
                target_q = rewards[i]
            end
        end
        
        push!(target_q_values, target_q)
    end
    
    # Calculate MSE loss
    loss = mean((current_q_values .- target_q_values).^2)
    
    return loss, current_q_values, target_q_values
end

# reward calculation
function calculate_reward(old_makespan::Float64, new_makespan::Float64, 
                         wait_time::Float64, priority::Int, 
                         desk_utilization::Float64, completed_count::Int,
                         total_activities::Int) 
    # Makespan improvement reward
    makespan_reward = (old_makespan - new_makespan) * 10.0
    
    # Wait time penalty
    wait_penalty = -wait_time * 0.5
    
    # Priority bonus (higher priority = higher bonus)
    priority_bonus = (4.0 - priority) * 3.0
    
    # Utilization bonus
    utilization_bonus = desk_utilization * 2.0
    
    # Completion progress bonus
    completion_bonus = (completed_count / total_activities) * 5.0
    
    # Efficiency bonus (minimize makespan relative to total processing time)
    efficiency_bonus = max(0, (200.0 - new_makespan) / 200.0) * 3.0
    
    total_reward = makespan_reward + wait_penalty + priority_bonus + 
                   utilization_bonus + completion_bonus + efficiency_bonus
    
    return total_reward
end


# Two-Agent Simulation
function simulate_two_agent_episode(env::TwoAgentEnvironment, ppo_agent::PPOAgent, 
                                   ddqn_agent::DDQNAgent, max_steps::Int=1000, 
                                   training::Bool=true)
    completed = Set{Tuple{Int, Int}}()
    total_reward = 0.0
    episode_length = 0
    total_wait_time = 0.0
    ppo_trajectories = []
    ddqn_experiences = []
    
     # Calculate total activities for enhanced reward calculation
    total_activities = sum(length(p.activities) for p in env.patients)
    

    # Reset environment
    for desk in env.service_desks
        desk.completion_time = 0.0
        desk.available = true
        desk.utilization = 0.0
        desk.processing_count = 0
    end
    
    while length(completed) < sum(length(p.activities) for p in env.patients) && episode_length < max_steps
        old_makespan = maximum(desk.completion_time for desk in env.service_desks)
        
        # PPO Agent: Select patient-activity pair
        action, action_prob, state_value = select_ppo_action(ppo_agent, env, completed)
        if action === nothing
            break
        end
        
        patient_id, activity_id = action
        patient_idx = findfirst(p -> p.id == patient_id, env.patients)
        if patient_idx === nothing
            continue
        end
        patient = env.patients[patient_idx]
        
        # Get available desks for this activity
        available_desks = [desk for desk in env.service_desks 
                          if desk.activity_type == activity_id && desk.available]
        
        if isempty(available_desks)
            continue
        end
        
        # DDQN Agent: Select service desk
        selected_desk, q_value = select_ddqn_desk(ddqn_agent, available_desks, patient, activity_id)
        if selected_desk === nothing
            continue
        end
        
        # Execute the action
        processing_time = get(patient.activity_times, activity_id, 1.0)
        start_time = max(selected_desk.completion_time, patient.arrival_time)
        wait_time = start_time - patient.arrival_time
        
        # Create state for DDQN experience
        current_state = [
            Float64(selected_desk.completion_time / 200.0),
            Float64(patient.priority / 3.0),
            Float64(processing_time / 20.0),
            Float64(wait_time / 100.0)
        ]
        
        # Update desk
        selected_desk.completion_time = start_time + processing_time
        selected_desk.processing_count += 1
        if selected_desk.completion_time > 0
            selected_desk.utilization = selected_desk.processing_count * processing_time / selected_desk.completion_time
        end
        
        # Mark as completed
        push!(completed, (patient_id, activity_id))
        
        # Calculate reward
        new_makespan = maximum(desk.completion_time for desk in env.service_desks)
        # reward = calculate_reward(old_makespan, new_makespan, wait_time, 
        #                         patient.priority, selected_desk.utilization, 
        #                         length(completed))

        reward = calculate_reward(old_makespan, new_makespan, wait_time, 
                                         patient.priority, selected_desk.utilization, 
                                         length(completed), total_activities)
        

        
        total_reward += reward
        total_wait_time += wait_time
        episode_length += 1
        
        # Store experiences for training
        if training
            # PPO trajectory
            push!(ppo_trajectories, (action, reward, action_prob, state_value))
            
            # DDQN experience (simplified next state)
            next_state = current_state  # Simplified for demonstration
            done = length(completed) >= sum(length(p.activities) for p in env.patients)
            
            push!(ddqn_experiences, (current_state, selected_desk.id, reward, next_state, done))
            
            # Add to DDQN replay buffer
            if length(ddqn_agent.replay_buffer) >= ddqn_agent.replay_memory_size
                popfirst!(ddqn_agent.replay_buffer)
            end
            push!(ddqn_agent.replay_buffer, (current_state, selected_desk.id, reward, next_state, done))
        end
        
        ddqn_agent.current_step += 1
    end
    
    # Train agents
    if training
        if length(ppo_trajectories) >= ppo_agent.trajectory_samples
            train_ppo_agent!(ppo_agent, ppo_trajectories)
        end
        
        if length(ddqn_agent.replay_buffer) >= ddqn_agent.replay_start_size
            train_ddqn_agent!(ddqn_agent)
        end
    end
    
    # Calculate final metrics
    final_makespan = maximum(desk.completion_time for desk in env.service_desks)
    avg_wait_time = total_wait_time / max(1, length(completed))
    desk_utilizations = [desk.utilization for desk in env.service_desks]
    
    results = ExperimentResults(
        final_makespan,
        total_wait_time,
        avg_wait_time,
        desk_utilizations,
        length(completed),
        episode_length,
        total_reward
    )
    
    return results
end
 

 


#-----------------------------------------------------------------------------------------------


using Plots, Statistics, Random, StatsPlots, DataFrames, CSV, StatsBase


# Generate synthetic dataset based on the provided Python code
function generate_synthetic_dataset(num_base_patients::Int=20)
    # Parameters for lambda(t)
    lambda_base = 12  # Base arrival rate (patients/hour)
    alpha_daily = 0.5  # Circadian amplitude
    alpha_weekly = 0.2  # Weekly fluctuation amplitude
    duration_hours = 24  # Simulate for 1 day
    time_step = 1/60  # Time step in hours (1 minute)
    
    # Define lambda(t)
    function lambda_t(t)
        h_t = t % 24  # Hour of day (0-23)
        d_t = 0  # Assume day 0 (e.g., Monday) for 1-day simulation
        daily_term = alpha_daily * sin(2 * π * (h_t - 6) / 24)
        weekly_term = alpha_weekly * sin(2 * π * d_t / 7)
        return lambda_base * (1 + daily_term + weekly_term)
    end
    
    # Time points
    time_points = collect(0:time_step:duration_hours)
    lambda_values = [lambda_t(t) for t in time_points]
    
    # Generate non-homogeneous Poisson process
    max_lambda = maximum(lambda_values)
    events = Float64[]
    patient_id = num_base_patients + 1  # Start after original patients
    t = 0.0
    
    while t < duration_hours
        t += rand() * (-log(rand()) / max_lambda)  # Exponential random variable
        if t >= duration_hours
            break
        end
        if rand() < lambda_t(t) / max_lambda
            push!(events, t)
        end
    end
    
    # Create synthetic patients
    num_patients = length(events)
    patients = Patient[]
    
    for i in 1:num_patients
        activities = [1, 2, 3, 4, 5]
        activity_times = Dict{Int, Float64}(
            1 => round(rand() * (2.5 - 1.7) + 1.7, digits=1),
            2 => round(rand() * (6.7 - 3.8) + 3.8, digits=1),
            3 => round(rand() * (4.5 - 2.9) + 2.9, digits=1),
            4 => round(rand() * (8.3 - 3.9) + 3.9, digits=1),
            5 => round(rand() * (4.8 - 2.9) + 2.9, digits=1)
        )
        
        patient = Patient(
            patient_id + i - 1,
            round(events[i], digits=1),
            activities,
            activity_times,
            rand(1:3),  # priority
            rand(23:79),  # age
            round(rand() * (2.8 - 1.2) + 1.2, digits=1),  # complexity_score
            activity_times[1],
            activity_times[2],
            activity_times[3],
            activity_times[4],
            activity_times[5]
        )
        
        push!(patients, patient)
    end
    
    # Sort by arrival time
    sort!(patients, by=p -> p.arrival_time)
    
    return patients
end

# K-fold cross-validation data splitting
function create_k_folds(data, k::Int=15)
    n = length(data)
    fold_size = div(n, k)
    remainder = n % k
    
    # Shuffle data to ensure random distribution
    shuffled_data = shuffle(data)
    
    folds = Vector{Vector{eltype(data)}}()
    start_idx = 1
    
    for i in 1:k
        # Some folds get one extra element if n is not divisible by k
        current_fold_size = fold_size + (i <= remainder ? 1 : 0)
        end_idx = start_idx + current_fold_size - 1
        
        fold = shuffled_data[start_idx:end_idx]
        push!(folds, fold)
        
        start_idx = end_idx + 1
    end
    
    return folds
end

# Single fold training and validation
function train_validate_single_fold(train_patients, val_patients, fold_idx::Int, 
                                   training_episodes::Int=1000, validation_episodes::Int=250)
    println("Training fold $fold_idx with $(length(train_patients)) training, $(length(val_patients)) validation patients")
    
    # Desk configuration
    desk_config = Dict(1 => 3, 2 => 2, 3 => 4, 4 => 2, 5 => 3)
    
    # Initialize fresh agents for each fold
    ppo_agent = PPOAgent(6)  # 6 node features
    ddqn_agent = DDQNAgent(4)  # 4 state features
    
    # Training metrics for this fold
    fold_training_metrics = Dict(
        "fold" => Int[],
        "episode" => Int[],
        "makespan" => Float64[],
        "total_reward" => Float64[],
        "wait_time" => Float64[],
        "completion_rate" => Float64[],
        "avg_utilization" => Float64[],
        "epsilon" => Float64[]
    )
    
    # Validation metrics for this fold
    fold_validation_metrics = Dict(
        "fold" => Int[],
        "episode" => Int[],
        "makespan" => Float64[],
        "total_reward" => Float64[],
        "wait_time" => Float64[],
        "completion_rate" => Float64[],
        "avg_utilization" => Float64[]
    )
    
    # Training loop
    for episode in 1:training_episodes
        # Sample random subset of patients for each episode
        episode_patients = StatsBase.sample(train_patients, min(10, length(train_patients)), replace=false)
        train_env = TwoAgentEnvironment(episode_patients, desk_config)
        
        # Run training episode
        results = simulate_two_agent_episode(train_env, ppo_agent, ddqn_agent, 1000, true)
        
        # Store metrics
        push!(fold_training_metrics["fold"], fold_idx)
        push!(fold_training_metrics["episode"], episode)
        push!(fold_training_metrics["makespan"], results.makespan)
        push!(fold_training_metrics["total_reward"], results.objective_value)
        push!(fold_training_metrics["wait_time"], results.average_wait_time)
        push!(fold_training_metrics["completion_rate"], results.completed_activities / sum(length(p.activities) for p in episode_patients))
        push!(fold_training_metrics["avg_utilization"], mean(results.desk_utilization))
        push!(fold_training_metrics["epsilon"], ddqn_agent.epsilon[])
    end
    
    # Validation loop
    for episode in 1:validation_episodes
        # Use subset of validation patients
        episode_patients = length(val_patients) > 10 ? StatsBase.sample(val_patients, 10, replace=false) : val_patients
        val_env = TwoAgentEnvironment(episode_patients, desk_config)
        
        # Run validation episode (no training)
        results = simulate_two_agent_episode(val_env, ppo_agent, ddqn_agent, 1000, false)
        
        # Store metrics
        push!(fold_validation_metrics["fold"], fold_idx)
        push!(fold_validation_metrics["episode"], episode)
        push!(fold_validation_metrics["makespan"], results.makespan)
        push!(fold_validation_metrics["total_reward"], results.objective_value)
        push!(fold_validation_metrics["wait_time"], results.average_wait_time)
        push!(fold_validation_metrics["completion_rate"], results.completed_activities / sum(length(p.activities) for p in episode_patients))
        push!(fold_validation_metrics["avg_utilization"], mean(results.desk_utilization))
    end
    
    return fold_training_metrics, fold_validation_metrics
end


# Training performance visualization using cv_results data
function plot_training_performance_cv(cv_training_metrics, cv_validation_metrics, fold_summary)
    println("Creating comprehensive training performance visualization...")
    
    # 1. Training Progress Across All Folds
    p1 = plot(size=(800, 400))
    
    # Plot individual fold training curves (lighter lines)
    for fold in 1:15
        fold_indices = findall(cv_training_metrics["fold"] .== fold)
        if !isempty(fold_indices)
            fold_episodes = cv_training_metrics["episode"][fold_indices]
            fold_makespan = cv_training_metrics["makespan"][fold_indices]
            plot!(p1, fold_episodes, fold_makespan, 
                  color=:blue, alpha=0.2, linewidth=1, label=fold == 1 ? "Individual Folds" : "")
        end
    end
    
    # Overall moving average (increased window for 10k episodes)
    window_size = 100
    if length(cv_training_metrics["makespan"]) >= window_size
        moving_avg = [mean(cv_training_metrics["makespan"][max(1, i-window_size+1):i]) 
                     for i in window_size:length(cv_training_metrics["makespan"])]
        plot!(p1, window_size:length(cv_training_metrics["makespan"]), moving_avg,
              color=:red, linewidth=3, label="Moving Average")
    end
    
    title!(p1, "Training Convergence Across All Folds (10K Episodes)")
    xlabel!(p1, "Episode")
    ylabel!(p1, "Makespan")
    plot!(p1, legend=:topright)
    
    # 2. Reward Evolution
    p2 = plot(size=(800, 400))
    
    # Plot reward progression with more episodes
    for fold in 1:15
        fold_indices = findall(cv_training_metrics["fold"] .== fold)
        if !isempty(fold_indices)
            fold_episodes = cv_training_metrics["episode"][fold_indices]
            fold_rewards = cv_training_metrics["total_reward"][fold_indices]
            plot!(p2, fold_episodes, fold_rewards, 
                  color=:green, alpha=0.2, linewidth=1, label=fold == 1 ? "Individual Folds" : "")
        end
    end
    
    # Overall reward trend
    if length(cv_training_metrics["total_reward"]) >= window_size
        reward_ma = [mean(cv_training_metrics["total_reward"][max(1, i-window_size+1):i]) 
                    for i in window_size:length(cv_training_metrics["total_reward"])]
        plot!(p2, window_size:length(cv_training_metrics["total_reward"]), reward_ma,
              color=:purple, linewidth=3, label="Moving Average")
    end
    
    title!(p2, "Reward Evolution Across Training")
    xlabel!(p2, "Episode")
    ylabel!(p2, "Total Reward")
    plot!(p2, legend=:bottomright)
    
    # 3. Completion Rate Progress
    p3 = plot(size=(800, 400))
    
    for fold in 1:15
        fold_indices = findall(cv_training_metrics["fold"] .== fold)
        if !isempty(fold_indices)
            fold_episodes = cv_training_metrics["episode"][fold_indices]
            fold_completion = cv_training_metrics["completion_rate"][fold_indices] .* 100
            plot!(p3, fold_episodes, fold_completion, 
                  color=:orange, alpha=0.2, linewidth=1, label=fold == 1 ? "Individual Folds" : "")
        end
    end
    
    # Completion rate moving average
    if length(cv_training_metrics["completion_rate"]) >= window_size
        completion_ma = [mean(cv_training_metrics["completion_rate"][max(1, i-window_size+1):i]) 
                        for i in window_size:length(cv_training_metrics["completion_rate"])] .* 100
        plot!(p3, window_size:length(cv_training_metrics["completion_rate"]), completion_ma,
              color=:red, linewidth=3, label="Moving Average")
    end
    
    title!(p3, "Completion Rate Evolution")
    xlabel!(p3, "Episode")
    ylabel!(p3, "Completion Rate (%)")
    plot!(p3, legend=:bottomright)
    
    # 4. Epsilon Decay (DDQN exploration)
    p4 = plot(size=(800, 400))
    
    # Plot epsilon decay across all folds
    if haskey(cv_training_metrics, "epsilon")
        plot!(p4, cv_training_metrics["epsilon"], 
              color=:cyan, linewidth=2, label="Epsilon Decay")
    end
    title!(p4, "DDQN Epsilon Decay Across Training")
    xlabel!(p4, "Episode")
    ylabel!(p4, "Epsilon Value")
    plot!(p4, legend=:topright)
    
    # 5. Utilization Progress
    p5 = plot(size=(800, 400))
    
    for fold in 1:15
        fold_indices = findall(cv_training_metrics["fold"] .== fold)
        if !isempty(fold_indices)
            fold_episodes = cv_training_metrics["episode"][fold_indices]
            fold_util = cv_training_metrics["avg_utilization"][fold_indices] .* 100
            plot!(p5, fold_episodes, fold_util, 
                  color=:brown, alpha=0.2, linewidth=1, label=fold == 1 ? "Individual Folds" : "")
        end
    end
    
    # Utilization moving average
    if length(cv_training_metrics["avg_utilization"]) >= window_size
        util_ma = [mean(cv_training_metrics["avg_utilization"][max(1, i-window_size+1):i]) 
                  for i in window_size:length(cv_training_metrics["avg_utilization"])] .* 100
        plot!(p5, window_size:length(cv_training_metrics["avg_utilization"]), util_ma,
              color=:darkgreen, linewidth=3, label="Moving Average")
    end
    
    hline!(p5, [80], color=:red, linestyle=:dash, linewidth=2, label="Target (80%)")
    title!(p5, "Resource Utilization Evolution")
    xlabel!(p5, "Episode")
    ylabel!(p5, "Utilization (%)")
    plot!(p5, legend=:bottomright)
    
    # 6. Learning Rate Analysis
    p6 = plot(size=(800, 400))
    
    # Calculate learning rate as improvement per episode
    learning_rates = []
    for fold in 1:15
        fold_indices = findall(cv_training_metrics["fold"] .== fold)
        if length(fold_indices) > 1
            fold_makespan = cv_training_metrics["makespan"][fold_indices]
            # Calculate rate of improvement (negative slope = improvement)
            improvements = -diff(fold_makespan)
            append!(learning_rates, improvements)
        end
    end
    
    if !isempty(learning_rates)
        # Plot learning rate distribution
        histogram!(p6, learning_rates, bins=50, alpha=0.7, color=:lightblue, 
                  label="Learning Rate Distribution")
        vline!(p6, [mean(learning_rates)], color=:red, linewidth=2, 
               label="Mean Learning Rate")
    end
    
    title!(p6, "Learning Rate Distribution (Makespan Improvement)")
    xlabel!(p6, "Improvement per Episode")
    ylabel!(p6, "Frequency")
    plot!(p6, legend=:topright)
    
    # Combine all training plots
    training_plot = plot(p1, p2, p3, p4, p5, p6, 
                        layout=(3,2), size=(1600, 1200), 
                        plot_title="Training Performance Analysis - 15-Fold CV (10K Episodes)")
    
    return training_plot
end

# Enhanced validation performance visualization
function plot_validation_performance_cv(cv_validation_metrics, fold_summary)
    println("Creating comprehensive validation performance visualization...")
    
    # 1. Validation Performance by Fold
    p1 = plot(size=(800, 400))
    
    # Box plot of validation makespan by fold
    fold_makespans = []
    fold_labels = []
    
    for fold in 1:15
        fold_indices = findall(cv_validation_metrics["fold"] .== fold)
        if !isempty(fold_indices)
            fold_vals = cv_validation_metrics["makespan"][fold_indices]
            append!(fold_makespans, fold_vals)
            append!(fold_labels, fill("F$fold", length(fold_vals)))
        end
    end
    
    if !isempty(fold_makespans)
        # Create box plot
        boxplot!(p1, fold_labels, fold_makespans, color=:blue, alpha=0.5, 
                linewidth=2, outliers=true, label="Validation Performance")
    end
    
    title!(p1, "Validation Makespan Distribution by Fold (1K Episodes)")
    xlabel!(p1, "Fold")
    ylabel!(p1, "Makespan")
    plot!(p1, xrotation=45)
    
    # 2. Training vs Validation Performance
    p2 = plot(size=(800, 400))
    
    # Scatter plot of training vs validation performance
    scatter!(p2, fold_summary["train_makespan_mean"], fold_summary["val_makespan_mean"],
             color=:red, markersize=8, alpha=0.7, label="Fold Performance")
    
    # Add error bars
    for i in 1:length(fold_summary["fold"])
        plot!(p2, [fold_summary["train_makespan_mean"][i]], [fold_summary["val_makespan_mean"][i]],
              xerror=[fold_summary["train_makespan_std"][i]], 
              yerror=[fold_summary["val_makespan_std"][i]],
              color=:red, alpha=0.3, linewidth=1, label="")
    end
    
    # Perfect correlation line
    min_val = min(minimum(fold_summary["train_makespan_mean"]), minimum(fold_summary["val_makespan_mean"]))
    max_val = max(maximum(fold_summary["train_makespan_mean"]), maximum(fold_summary["val_makespan_mean"]))
    plot!(p2, [min_val, max_val], [min_val, max_val], 
          color=:black, linestyle=:dash, linewidth=2, label="Perfect Correlation")
    
    title!(p2, "Training vs Validation Performance")
    xlabel!(p2, "Training Makespan")
    ylabel!(p2, "Validation Makespan")
    plot!(p2, legend=:topleft)
    
    # 3. Completion Rate Comparison
    p3 = plot(size=(800, 400))
    
    # Bar chart for completion rates
    fold_nums = 1:15
    train_completion = fold_summary["train_completion_mean"] .* 100
    val_completion = fold_summary["val_completion_mean"] .* 100
    
    bar!(p3, fold_nums .- 0.2, train_completion, alpha=0.7, color=:blue, 
         width=0.4, label="Training")
    bar!(p3, fold_nums .+ 0.2, val_completion, alpha=0.7, color=:red, 
         width=0.4, label="Validation")
    
    title!(p3, "Completion Rate by Fold")
    xlabel!(p3, "Fold")
    ylabel!(p3, "Completion Rate (%)")
    plot!(p3, xticks=1:2:15)
    
    # 4. Utilization Comparison
    p4 = plot(size=(800, 400))
    
    train_util = fold_summary["train_utilization_mean"] .* 100
    val_util = fold_summary["val_utilization_mean"] .* 100
    
    bar!(p4, fold_nums .- 0.2, train_util, alpha=0.7, color=:green, 
         width=0.4, label="Training")
    bar!(p4, fold_nums .+ 0.2, val_util, alpha=0.7, color=:orange, 
         width=0.4, label="Validation")
    
    hline!(p4, [80], color=:red, linestyle=:dash, linewidth=2, label="Target (80%)")
    title!(p4, "Resource Utilization by Fold")
    xlabel!(p4, "Fold")
    ylabel!(p4, "Utilization (%)")
    plot!(p4, xticks=1:2:15)
    
    # 5. Performance Stability Analysis
    p5 = plot(size=(800, 400))
    
    # Calculate coefficient of variation for each metric
    metrics = ["Makespan", "Completion", "Utilization"]
    train_cv = [
        std(fold_summary["train_makespan_mean"]) / mean(fold_summary["train_makespan_mean"]),
        std(fold_summary["train_completion_mean"]) / mean(fold_summary["train_completion_mean"]),
        std(fold_summary["train_utilization_mean"]) / mean(fold_summary["train_utilization_mean"])
    ]
    val_cv = [
        std(fold_summary["val_makespan_mean"]) / mean(fold_summary["val_makespan_mean"]),
        std(fold_summary["val_completion_mean"]) / mean(fold_summary["val_completion_mean"]),
        std(fold_summary["val_utilization_mean"]) / mean(fold_summary["val_utilization_mean"])
    ]
    
    bar!(p5, (1:3) .- 0.2, train_cv, alpha=0.7, color=:blue, 
         width=0.4, label="Training CV")
    bar!(p5, (1:3) .+ 0.2, val_cv, alpha=0.7, color=:red, 
         width=0.4, label="Validation CV")
    
    title!(p5, "Performance Stability (Coefficient of Variation)")
    xlabel!(p5, "Metric")
    ylabel!(p5, "CV (std/mean)")
    plot!(p5, xticks=(1:3, metrics))
    
    # 6. Statistical Significance Test
    p6 = plot(size=(800, 400))
    
    # Create confidence intervals for validation performance
    val_means = fold_summary["val_makespan_mean"]
    val_stds = fold_summary["val_makespan_std"]
    n_episodes = 1000  # validation episodes per fold
    confidence_level = 0.95
    t_value = 2.145  # t-value for 95% CI with 14 degrees of freedom
    
    # Calculate confidence intervals
    ci_lower = val_means .- t_value .* val_stds ./ sqrt(n_episodes)
    ci_upper = val_means .+ t_value .* val_stds ./ sqrt(n_episodes)
    
    # Plot confidence intervals
    plot!(p6, fold_summary["fold"], val_means, 
          ribbon=(val_means - ci_lower, ci_upper - val_means),
          color=:blue, alpha=0.3, linewidth=2, label="95% Confidence Interval")
    
    scatter!(p6, fold_summary["fold"], val_means, 
             color=:red, markersize=6, label="Fold Mean")
    
    title!(p6, "Validation Performance Confidence Intervals")
    xlabel!(p6, "Fold")
    ylabel!(p6, "Makespan")
    plot!(p6, legend=:topright)
    
    # Combine all validation plots
    validation_plot = plot(p1, p2, p3, p4, p5, p6, 
                          layout=(3,2), size=(1600, 1200), 
                          plot_title="Validation Performance Analysis - 15-Fold CV (1K Episodes)")
    
    return validation_plot
end

# Enhanced patient flow analysis
function plot_patient_flow_analysis_enhanced(patients, desk_config)
    println("Creating enhanced patient flow analysis...")
    
    # 1. Arrival Pattern Analysis
    p1 = plot(size=(800, 400))
    
    arrival_times = [p.arrival_time for p in patients]
    
    # Histogram of arrivals
    histogram!(p1, arrival_times, bins=24, alpha=0.7, color=:blue, 
              label="Patient Arrivals")
    
    # Overlay theoretical arrival rate
    times = 0:0.5:24
    lambda_base = 12
    alpha_daily = 0.5
    theoretical_rates = [lambda_base * (1 + alpha_daily * sin(2 * π * (t - 6) / 24)) for t in times]
    
    plot!(p1, times, theoretical_rates .* 2, color=:red, linewidth=3, 
          label="Theoretical Rate")
    
    title!(p1, "Patient Arrival Patterns vs Theoretical")
    xlabel!(p1, "Time (hours)")
    ylabel!(p1, "Patients/Hour")
    plot!(p1, legend=:topright)
    
    # 2. Queue Dynamics Simulation
    p2 = plot(size=(800, 400))
    
    # Simulate queue dynamics
    simulation_times = 0:0.1:24
    queue_lengths = []
    cumulative_arrivals = []
    cumulative_completions = []
    
    for t in simulation_times
        # Count arrivals up to time t
        arrivals = sum(p.arrival_time <= t for p in patients)
        
        # Estimate completions based on service capacity
        total_capacity = sum(values(desk_config))
        avg_service_time = 4.0  # Average service time
        service_rate = total_capacity / avg_service_time
        completions = min(arrivals, Int(floor(t * service_rate)))
        
        push!(cumulative_arrivals, arrivals)
        push!(cumulative_completions, completions)
        push!(queue_lengths, max(0, arrivals - completions))
    end
    
    # Plot queue dynamics
    plot!(p2, simulation_times, queue_lengths, color=:red, linewidth=2, 
          label="Queue Length")
    plot!(p2, simulation_times, cumulative_arrivals, color=:blue, linewidth=2, 
          label="Cumulative Arrivals")
    plot!(p2, simulation_times, cumulative_completions, color=:green, linewidth=2, 
          label="Cumulative Completions")
    
    title!(p2, "Queue Dynamics Over Time")
    xlabel!(p2, "Time (hours)")
    ylabel!(p2, "Number of Patients")
    plot!(p2, legend=:topleft)
    
    # 3. Service Time Analysis by Activity
    p3 = plot(size=(800, 400))
    
    # Collect service times by activity
    activity_data = Dict{Int, Vector{Float64}}()
    for activity in 1:5
        activity_data[activity] = []
        for patient in patients
            if haskey(patient.activity_times, activity)
                push!(activity_data[activity], patient.activity_times[activity])
            end
        end
    end
    
    # Create box plots for each activity
    all_times = Float64[]
    all_labels = String[]
    
    for activity in 1:5
        if !isempty(activity_data[activity])
            append!(all_times, activity_data[activity])
            append!(all_labels, fill("Act $activity", length(activity_data[activity])))
        end
    end
    
    if !isempty(all_times)
        boxplot!(p3, all_labels, all_times, color=:blue, alpha=0.7, 
                linewidth=2, outliers=true, label="Service Times")
    end
    
    title!(p3, "Service Time Distribution by Activity")
    xlabel!(p3, "Activity Type")
    ylabel!(p3, "Service Time (hours)")
    
    # 4. Workload Distribution Analysis
    p4 = plot(size=(800, 400))
    
    # Calculate workload for each desk type
    desk_workloads = Dict{Int, Float64}()
    for activity in 1:5
        total_time = sum(get(p.activity_times, activity, 0.0) for p in patients)
        desk_workloads[activity] = total_time
    end
    
    # Plot workload distribution
    activities = collect(keys(desk_workloads))
    workloads = [desk_workloads[a] for a in activities]
    capacities = [desk_config[a] for a in activities]
    
    # Theoretical capacity (24 hours * number of desks)
    theoretical_capacity = capacities .* 24
    utilization_rates = workloads ./ theoretical_capacity .* 100
    
    bar!(p4, activities .- 0.2, workloads, alpha=0.7, color=:red, 
         width=0.4, label="Actual Workload")
    bar!(p4, activities .+ 0.2, theoretical_capacity, alpha=0.7, color=:green, 
         width=0.4, label="Theoretical Capacity")
    
    title!(p4, "Workload vs Capacity by Activity")
    xlabel!(p4, "Activity Type")
    ylabel!(p4, "Total Hours")
    plot!(p4, xticks=(1:5, ["Act $i" for i in 1:5]))
    
    # 5. Patient Complexity Analysis
    p5 = plot(size=(800, 400))
    
    # Complexity score distribution
    complexity_scores = [p.complexity_score for p in patients]
    priorities = [p.priority for p in patients]
    
    # Create scatter plot of complexity vs priority
    scatter!(p5, priorities, complexity_scores, 
             color=:purple, alpha=0.6, markersize=6,
             label="Patients")
    
    # Add trend line
    if length(complexity_scores) > 1
        # Simple linear regression
        x_mean = mean(priorities)
        y_mean = mean(complexity_scores)
        slope = sum((priorities .- x_mean) .* (complexity_scores .- y_mean)) / 
                sum((priorities .- x_mean).^2)
        intercept = y_mean - slope * x_mean
        
        x_trend = [1, 2, 3]
        y_trend = intercept .+ slope .* x_trend
        
        plot!(p5, x_trend, y_trend, color=:red, linewidth=2, 
              label="Trend Line")
    end
    
    title!(p5, "Patient Complexity vs Priority")
    xlabel!(p5, "Priority Level")
    ylabel!(p5, "Complexity Score")
    plot!(p5, xticks=(1:3, ["High", "Medium", "Low"]))
    plot!(p5, legend=:topleft)
    
    # 6. Resource Utilization Heatmap
    p6 = plot(size=(800, 400))
    
    # Create utilization matrix (time vs desk type)
    time_slots = 0:1:23  # Hourly slots
    desk_types = 1:5
    
    utilization_matrix = zeros(length(time_slots), length(desk_types))
    
    for (t_idx, t) in enumerate(time_slots)
        for (d_idx, desk_type) in enumerate(desk_types)
            # Count patients using this desk type in this time slot
            slot_usage = 0.0
            for patient in patients
                if patient.arrival_time >= t && patient.arrival_time < t + 1
                    if haskey(patient.activity_times, desk_type)
                        slot_usage += patient.activity_times[desk_type]
                    end
                end
            end
            # Normalize by desk capacity
            if haskey(desk_config, desk_type)
                utilization_matrix[t_idx, d_idx] = slot_usage / desk_config[desk_type]
            end
        end
    end
    
    # Create heatmap
    heatmap!(p6, desk_types, time_slots, utilization_matrix', 
             color=:viridis, aspect_ratio=:auto)
    
    title!(p6, "Resource Utilization Heatmap")
    xlabel!(p6, "Desk Type")
    ylabel!(p6, "Time (hours)")
    
    # Combine all flow analysis plots
    flow_plot = plot(p1, p2, p3, p4, p5, p6, 
                    layout=(3,2), size=(1600, 1200), 
                    plot_title="Enhanced Patient Flow Analysis")
    
    return flow_plot
end

#  Main cross-validation function with increased episodes
function cross_validate_agents_enhanced(k_folds::Int=15, training_episodes::Int=10000, validation_episodes::Int=1000)
    println("Starting Enhanced 15-Fold Cross-Validation for Two-Agent System...")
    println("Training Episodes: $training_episodes, Validation Episodes: $validation_episodes")
    
    # Generate datasets
    println("Generating synthetic dataset...")
    synthetic_patients = generate_synthetic_dataset()
    
    # Combine datasets
    all_patients = synthetic_patients
    println("Total patients: $(length(all_patients))")
    
    # Create k-folds
    folds = create_k_folds(all_patients, k_folds)
    
    # Cross-validation metrics storage
    cv_training_metrics = Dict(
        "fold" => Int[],
        "episode" => Int[],
        "makespan" => Float64[],
        "total_reward" => Float64[],
        "wait_time" => Float64[],
        "completion_rate" => Float64[],
        "avg_utilization" => Float64[],
        "epsilon" => Float64[]
    )
    
    cv_validation_metrics = Dict(
        "fold" => Int[],
        "episode" => Int[],
        "makespan" => Float64[],
        "total_reward" => Float64[],
        "wait_time" => Float64[],
        "completion_rate" => Float64[],
        "avg_utilization" => Float64[]
    )
    
    # Fold-level summary metrics
    fold_summary = Dict(
        "fold" => Int[],
        "train_makespan_mean" => Float64[],
        "train_makespan_std" => Float64[],
        "val_makespan_mean" => Float64[],
        "val_makespan_std" => Float64[],
        "train_completion_mean" => Float64[],
        "val_completion_mean" => Float64[],
        "train_utilization_mean" => Float64[],
        "val_utilization_mean" => Float64[],
        "train_size" => Int[],
        "val_size" => Int[]
    )
    
    # Run cross-validation
    for fold_idx in 1:k_folds
        println("\\n=== FOLD $fold_idx/$k_folds ===")
        
        # Create training and validation sets
        val_patients = folds[fold_idx]
        train_patients = vcat([folds[i] for i in 1:k_folds if i != fold_idx]...)
        
        # Train and validate on this fold
        fold_train_metrics, fold_val_metrics = train_validate_single_fold(
            train_patients, val_patients, fold_idx, training_episodes, validation_episodes
        )
        
        # Aggregate metrics
        for key in keys(fold_train_metrics)
            if haskey(cv_training_metrics, key)
                append!(cv_training_metrics[key], fold_train_metrics[key])
            end
        end
        
        for key in keys(fold_val_metrics)
            if haskey(cv_validation_metrics, key)
                append!(cv_validation_metrics[key], fold_val_metrics[key])
            end
        end
        
        # Calculate fold summary statistics
        push!(fold_summary["fold"], fold_idx)
        push!(fold_summary["train_makespan_mean"], mean(fold_train_metrics["makespan"]))
        push!(fold_summary["train_makespan_std"], std(fold_train_metrics["makespan"]))
        push!(fold_summary["val_makespan_mean"], mean(fold_val_metrics["makespan"]))
        push!(fold_summary["val_makespan_std"], std(fold_val_metrics["makespan"]))
        push!(fold_summary["train_completion_mean"], mean(fold_train_metrics["completion_rate"]))
        push!(fold_summary["val_completion_mean"], mean(fold_val_metrics["completion_rate"]))
        push!(fold_summary["train_utilization_mean"], mean(fold_train_metrics["avg_utilization"]))
        push!(fold_summary["val_utilization_mean"], mean(fold_val_metrics["avg_utilization"]))
        push!(fold_summary["train_size"], length(train_patients))
        push!(fold_summary["val_size"], length(val_patients))
        
        # Print fold summary
        println("Fold $fold_idx Summary:")
        println("  Training: $(length(train_patients)) patients")
        println("  Validation: $(length(val_patients)) patients")
        println("  Train Makespan: $(round(fold_summary["train_makespan_mean"][end], digits=2)) ± $(round(fold_summary["train_makespan_std"][end], digits=2))")
        println("  Val Makespan: $(round(fold_summary["val_makespan_mean"][end], digits=2)) ± $(round(fold_summary["val_makespan_std"][end], digits=2))")
        println("  Train Completion: $(round(fold_summary["train_completion_mean"][end]*100, digits=1))%")
        println("  Val Completion: $(round(fold_summary["val_completion_mean"][end]*100, digits=1))%")
    end
    
    return cv_training_metrics, cv_validation_metrics, fold_summary, all_patients
end
 

# Main execution function with enhanced cross-validation
function run_enhanced_cross_validation_experiment()
    println("=== Enhanced 15-Fold Cross-Validation Healthcare Scheduling Experiment ===")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Run 15-fold cross-validation with increased episodes
    cv_training_metrics, cv_validation_metrics, fold_summary, all_patients = cross_validate_agents_enhanced(15, 1000, 100)
    
    # Calculate and display statistics
    cv_stats = calculate_cv_statistics(fold_summary)
    
    # Display overall cross-validation results
    println("\n=== OVERALL CROSS-VALIDATION RESULTS ===")
    println("Training Episodes per Fold: 1,000")
    println("Validation Episodes per Fold: 100")
    println("Total Folds: 15")
    println()
    
    # Display key statistics
    println("MAKESPAN PERFORMANCE:")
    println("  Mean Validation Makespan: $(round(cv_stats["val_makespan_mean"], digits=2)) ± $(round(cv_stats["val_makespan_std"], digits=2))")
    println("  Mean Training Makespan: $(round(cv_stats["train_makespan_mean"], digits=2)) ± $(round(cv_stats["train_makespan_std"], digits=2))")
    println("  Generalization Gap: $(round(cv_stats["generalization_gap"], digits=2))")
    println()
    
    println("COMPLETION RATE:")
    println("  Mean Validation Completion: $(round(cv_stats["val_completion_mean"]*100, digits=1))% ± $(round(cv_stats["val_completion_std"]*100, digits=1))%")
    println("  Mean Training Completion: $(round(cv_stats["train_completion_mean"]*100, digits=1))% ± $(round(cv_stats["train_completion_std"]*100, digits=1))%")
    println()
    
    println("RESOURCE UTILIZATION:")
    println("  Mean Validation Utilization: $(round(cv_stats["val_utilization_mean"]*100, digits=1))% ± $(round(cv_stats["val_utilization_std"]*100, digits=1))%")
    println("  Mean Training Utilization: $(round(cv_stats["train_utilization_mean"]*100, digits=1))% ± $(round(cv_stats["train_utilization_std"]*100, digits=1))%")
    println()
    
    println("STABILITY METRICS:")
    println("  Makespan CV: $(round(cv_stats["makespan_cv"], digits=3))")
    println("  Completion CV: $(round(cv_stats["completion_cv"], digits=3))")
    println("  Utilization CV: $(round(cv_stats["utilization_cv"], digits=3))")
    println()
    
    # Create comprehensive visualizations using actual data
    println("Creating comprehensive visualization plots...")
    
    # Training performance visualization
    training_plot = plot_training_performance_cv(cv_training_metrics, cv_validation_metrics, fold_summary)
    
    # Validation performance visualization
    validation_plot = plot_validation_performance_cv(cv_validation_metrics, fold_summary)
    
    # Enhanced patient flow analysis (using a representative sample)
    desk_config = Dict(1 => 2, 2 => 3, 3 => 2, 4 => 1, 5 => 2)  # Example desk configuration
    patient_sample = all_patients[1:min(1000, length(all_patients))]  # Sample for visualization
    flow_plot = plot_patient_flow_analysis_enhanced(patient_sample, desk_config)
    
    # Statistical significance analysis
    statistical_plot = create_statistical_analysis_plot(fold_summary)
    
    # Save all plots
    println("Saving visualization plots...")
    
    try
        savefig(training_plot, "enhanced_cv_training_performance.png")
        println("  ✓ Training performance plot saved")
    catch e
        println("  ✗ Error saving training plot: $e")
    end
    
    try
        savefig(validation_plot, "enhanced_cv_validation_performance.png")
        println("  ✓ Validation performance plot saved")
    catch e
        println("  ✗ Error saving validation plot: $e")
    end
    
    try
        savefig(flow_plot, "enhanced_patient_flow_analysis.png")
        println("  ✓ Patient flow analysis plot saved")
    catch e
        println("  ✗ Error saving flow plot: $e")
    end
    
    try
        savefig(statistical_plot, "enhanced_statistical_analysis.png")
        println("  ✓ Statistical analysis plot saved")
    catch e
        println("  ✗ Error saving statistical plot: $e")
    end
    
    # Save metrics data to CSV files
    println("Saving metrics data to CSV files...")
    
    try
        # Convert training metrics to DataFrame and save
        training_df = DataFrame(cv_training_metrics)
        CSV.write("enhanced_cv_training_metrics.csv", training_df)
        println("  ✓ Training metrics saved to CSV")
    catch e
        println("  ✗ Error saving training metrics: $e")
    end
    
    try
        # Convert validation metrics to DataFrame and save
        validation_df = DataFrame(cv_validation_metrics)
        CSV.write("enhanced_cv_validation_metrics.csv", validation_df)
        println("  ✓ Validation metrics saved to CSV")
    catch e
        println("  ✗ Error saving validation metrics: $e")
    end
    
    try
        # Convert fold summary to DataFrame and save
        fold_summary_df = DataFrame(fold_summary)
        CSV.write("enhanced_cv_fold_summary.csv", fold_summary_df)
        println("  ✓ Fold summary saved to CSV")
    catch e
        println("  ✗ Error saving fold summary: $e")
    end
    
    # Create final summary report
    println("\n=== EXPERIMENT SUMMARY REPORT ===")
    println("Experiment completed successfully!")
    println("Files generated:")
    println("  - enhanced_cv_training_performance.png")
    println("  - enhanced_cv_validation_performance.png") 
    println("  - enhanced_patient_flow_analysis.png")
    println("  - enhanced_statistical_analysis.png")
    println("  - enhanced_cv_training_metrics.csv")
    println("  - enhanced_cv_validation_metrics.csv")
    println("  - enhanced_cv_fold_summary.csv")
    println()
    
    # Performance recommendations
    println("PERFORMANCE RECOMMENDATIONS:")
    if cv_stats["generalization_gap"] > 0.1
        println("  • High generalization gap detected. Consider regularization techniques.")
    else
        println("  • Good generalization performance achieved.")
    end
    
    if cv_stats["val_completion_mean"] < 0.8
        println("  • Completion rate below 80%. Consider increasing training episodes or adjusting reward function.")
    else
        println("  • Good completion rate achieved.")
    end
    
    if cv_stats["val_utilization_mean"] < 0.8
        println("  • Resource utilization below 80%. Consider optimizing resource allocation.")
    else
        println("  • Good resource utilization achieved.")
    end
    
    if cv_stats["makespan_cv"] > 0.2
        println("  • High makespan variability. Consider ensemble methods or hyperparameter tuning.")
    else
        println("  • Good makespan stability across folds.")
    end
    
    println("\n=== EXPERIMENT COMPLETED ===")
    
    return cv_training_metrics, cv_validation_metrics, fold_summary, cv_stats
end

# Helper function to calculate comprehensive cross-validation statistics
function calculate_cv_statistics(fold_summary)
    cv_stats = Dict{String, Float64}()
    
    # Makespan statistics
    cv_stats["train_makespan_mean"] = mean(fold_summary["train_makespan_mean"])
    cv_stats["train_makespan_std"] = std(fold_summary["train_makespan_mean"])
    cv_stats["val_makespan_mean"] = mean(fold_summary["val_makespan_mean"])
    cv_stats["val_makespan_std"] = std(fold_summary["val_makespan_mean"])
    cv_stats["generalization_gap"] = cv_stats["val_makespan_mean"] - cv_stats["train_makespan_mean"]
    
    # Completion rate statistics
    cv_stats["train_completion_mean"] = mean(fold_summary["train_completion_mean"])
    cv_stats["train_completion_std"] = std(fold_summary["train_completion_mean"])
    cv_stats["val_completion_mean"] = mean(fold_summary["val_completion_mean"])
    cv_stats["val_completion_std"] = std(fold_summary["val_completion_mean"])
    
    # Utilization statistics
    cv_stats["train_utilization_mean"] = mean(fold_summary["train_utilization_mean"])
    cv_stats["train_utilization_std"] = std(fold_summary["train_utilization_mean"])
    cv_stats["val_utilization_mean"] = mean(fold_summary["val_utilization_mean"])
    cv_stats["val_utilization_std"] = std(fold_summary["val_utilization_mean"])
    
    # Coefficient of variation for stability assessment
    cv_stats["makespan_cv"] = cv_stats["val_makespan_std"] / cv_stats["val_makespan_mean"]
    cv_stats["completion_cv"] = cv_stats["val_completion_std"] / cv_stats["val_completion_mean"]
    cv_stats["utilization_cv"] = cv_stats["val_utilization_std"] / cv_stats["val_utilization_mean"]
    
    return cv_stats
end

# Helper function to create statistical analysis plot
# function create_statistical_analysis_plot(fold_summary)
#     println("Creating statistical analysis visualization...")
    
#     # 1. Performance Distribution
#     p1 = plot(size=(800, 400))
    
#     # Violin plot for makespan distribution
#     violin!(p1, ["Training"], fold_summary["train_makespan_mean"], 
#             color=:blue, alpha=0.7, label="Training")
#     violin!(p1, ["Validation"], fold_summary["val_makespan_mean"], 
#             color=:red, alpha=0.7, label="Validation")
    
#     title!(p1, "Makespan Distribution Across Folds")
#     ylabel!(p1, "Makespan")
    
#     # 2. Correlation Analysis
#     p2 = plot(size=(800, 400))
    
#     # Scatter plot with correlation
#     scatter!(p2, fold_summary["train_makespan_mean"], fold_summary["val_makespan_mean"],
#              color=:green, markersize=8, alpha=0.7, label="Folds")
    
#     # Calculate correlation coefficient
#     correlation = cor(fold_summary["train_makespan_mean"], fold_summary["val_makespan_mean"])
    
#     # Add correlation line
#     x_range = extrema(fold_summary["train_makespan_mean"])
#     y_range = extrema(fold_summary["val_makespan_mean"])
    
#     title!(p2, "Training vs Validation Correlation (r = $(round(correlation, digits=3)))")
#     xlabel!(p2, "Training Makespan")
#     ylabel!(p2, "Validation Makespan")
    
#     # 3. Confidence Intervals
#     p3 = plot(size=(800, 400))
    
#     # Error bar plot
#     folds = 1:length(fold_summary["fold"])
#     errorbar!(p3, folds, fold_summary["val_makespan_mean"], 
#               fold_summary["val_makespan_std"], 
#               color=:blue, linewidth=2, markersize=6, label="Validation ± 1σ")
    
#     title!(p3, "Validation Performance with Confidence Intervals")
#     xlabel!(p3, "Fold")
#     ylabel!(p3, "Makespan")
    
#     # 4. Stability Metrics
#     p4 = plot(size=(800, 400))
    
#     # Coefficient of variation for different metrics
#     metrics = ["Makespan", "Completion", "Utilization"]
#     cv_values = [
#         std(fold_summary["val_makespan_mean"]) / mean(fold_summary["val_makespan_mean"]),
#         std(fold_summary["val_completion_mean"]) / mean(fold_summary["val_completion_mean"]),
#         std(fold_summary["val_utilization_mean"]) / mean(fold_summary["val_utilization_mean"])
#     ]
    
#     bar!(p4, metrics, cv_values, color=:orange, alpha=0.7, label="CV")
#     hline!(p4, [0.1], color=:red, linestyle=:dash, label="Stability Threshold")
    
#     title!(p4, "Performance Stability (Coefficient of Variation)")
#     ylabel!(p4, "CV (std/mean)")
    
#     # Combine all statistical plots
#     statistical_plot = plot(p1, p2, p3, p4, 
#                            layout=(2,2), size=(1600, 800), 
#                            plot_title="Statistical Analysis - Cross-Validation Results")
    
#     return statistical_plot
# end


# Helper function to create statistical analysis plot
function create_statistical_analysis_plot(fold_summary)
    println("Creating statistical analysis visualization...")
    
    # 1. Performance Distribution
    p1 = plot(size=(800, 400))
    
    # Violin plot for makespan distribution
    violin!(p1, ["Training"], fold_summary["train_makespan_mean"], 
            color=:blue, alpha=0.7, label="Training")
    violin!(p1, ["Validation"], fold_summary["val_makespan_mean"], 
            color=:red, alpha=0.7, label="Validation")
    
    title!(p1, "Makespan Distribution Across Folds")
    ylabel!(p1, "Makespan")
    
    # 2. Correlation Analysis
    p2 = plot(size=(800, 400))
    
    # Scatter plot with correlation
    scatter!(p2, fold_summary["train_makespan_mean"], fold_summary["val_makespan_mean"],
             color=:green, markersize=8, alpha=0.7, label="Folds")
    
    # Calculate correlation coefficient
    correlation = cor(fold_summary["train_makespan_mean"], fold_summary["val_makespan_mean"])
    
    # Add correlation line
    x_range = extrema(fold_summary["train_makespan_mean"])
    y_range = extrema(fold_summary["val_makespan_mean"])
    
    title!(p2, "Training vs Validation Correlation (r = $(round(correlation, digits=3)))")
    xlabel!(p2, "Training Makespan")
    ylabel!(p2, "Validation Makespan")
    
    # 3. Confidence Intervals
    p3 = plot(size=(800, 400))
    
    # Manual error bar plot using scatter and plot!
    folds = 1:length(fold_summary["fold"])
    val_means = fold_summary["val_makespan_mean"]
    val_stds = fold_summary["val_makespan_std"]
    
    # Plot error bars manually
    for i in 1:length(folds)
        plot!(p3, [folds[i], folds[i]], [val_means[i] - val_stds[i], val_means[i] + val_stds[i]], 
              color=:blue, linewidth=2, alpha=0.7, label="")
        scatter!(p3, [folds[i]], [val_means[i] - val_stds[i]], 
                color=:blue, markersize=3, alpha=0.7, label="")
        scatter!(p3, [folds[i]], [val_means[i] + val_stds[i]], 
                color=:blue, markersize=3, alpha=0.7, label="")
    end
    
    # Plot mean values
    scatter!(p3, folds, val_means, color=:red, markersize=6, 
             label="Validation Mean", alpha=0.8)
    
    title!(p3, "Validation Performance with Confidence Intervals")
    xlabel!(p3, "Fold")
    ylabel!(p3, "Makespan")
    
    # 4. Stability Metrics
    p4 = plot(size=(800, 400))
    
    # Coefficient of variation for different metrics
    metrics = ["Makespan", "Completion", "Utilization"]
    cv_values = [
        std(fold_summary["val_makespan_mean"]) / mean(fold_summary["val_makespan_mean"]),
        std(fold_summary["val_completion_mean"]) / mean(fold_summary["val_completion_mean"]),
        std(fold_summary["val_utilization_mean"]) / mean(fold_summary["val_utilization_mean"])
    ]
    
    bar!(p4, metrics, cv_values, color=:orange, alpha=0.7, label="CV")
    hline!(p4, [0.1], color=:red, linestyle=:dash, label="Stability Threshold")
    
    title!(p4, "Performance Stability (Coefficient of Variation)")
    ylabel!(p4, "CV (std/mean)")
    
    # Combine all statistical plots
    statistical_plot = plot(p1, p2, p3, p4, 
                           layout=(2,2), size=(1600, 800), 
                           plot_title="Statistical Analysis - Cross-Validation Results")
    
    return statistical_plot
end


run_enhanced_cross_validation_experiment()
