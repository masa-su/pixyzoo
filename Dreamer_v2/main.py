import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, ATARI_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import CategoricalActorModel, bottle_tuple, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel
from planner import MPCPlanner
from utils import lineplot, imagine_ahead, lambda_return, FreezeParameters
from tensorboardX import SummaryWriter

from schedulers import init_scheduler


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer',
                    help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS +
                    CONTROL_SUITE_ENVS + ATARI_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true',
                    help='Symbolic features')
parser.add_argument('--max-episode-length', type=int,
                    default=1000, metavar='T', help='Max episode length')
# Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--experience-size', type=int,
                    default=1000000, metavar='D', help='Experience replay size')
parser.add_argument('--cnn-activation-function', type=str, default='relu',
                    choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu',
                    choices=dir(F), help='Model activation function a dense layer')
# Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--embedding-size', type=int, default=1536,
                    metavar='E', help='Observation embedding size')
parser.add_argument('--hidden-size', type=int, default=600,
                    metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=600,
                    metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=32,
                    metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2,
                    metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float,
                    default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000,
                    metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5,
                    metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int,
                    default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50,
                    metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50,
                    metavar='L', help='Chunk size')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true',
                    help='use LogProb loss for observation_model and reward_model training')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D',
                    help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1',
                    help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1',
                    help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0,
                    metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3,
                    metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5,
                    metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float,
                    default=2e-4, metavar='α', help='Learning rate')
parser.add_argument('--actor_learning-rate', type=float,
                    default=4e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_learning-rate', type=float,
                    default=1e-4, metavar='α', help='Learning rate')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                    help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
parser.add_argument('--adam-epsilon', type=float, default=1e-7,
                    metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float,
                    default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15,
                    metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99,
                    metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95,
                    metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10,
                    metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000,
                    metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100,
                    metavar='K', help='Number of top candidates to fit')
parser.add_argument('--disable_gru_norm', action='store_true',
                    help='Disable normalization of gru cells')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25,
                    metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10,
                    metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50,
                    metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience',
                    action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='',
                    metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='',
                    metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
parser.add_argument('--kl-free', type=str, default='0.0',
                    help='minimum kl loss')
parser.add_argument('--kl-balance', type=str, default='0.0', help='')
parser.add_argument('--kl-scale', type=str, default='0.0', help='')
parser.add_argument('--imag-gradient-mix', type=str, default='linear(0.1,0,2.5e6)',
                    help="ratio between reinforce grad and dynamics backprop grad")
parser.add_argument('--actor-entropy', type=str, default='linear(3e-3,3e-4,2.5e6)',
                    help='coefficient of actor entropy')
parser.add_argument('--actor-state-entropy', type=str, default='0',
                    help='coefficient of entropy of the latent state model')
parser.add_argument('--num-actor-layers', type=int, default=4,
                    help="number of the fc layers of e actor")
parser.add_argument('--num-actor-units', type=int, default=400,
                    help='number of hidden units in the each layer of the actor')
args = parser.parse_args()
# Overshooting distance cannot be greater than chunk size
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
    args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
           'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed,
          args.max_episode_length, args.action_repeat, args.bit_depth)
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * \
        D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    #TODO: fix replay buffer(env.action_sizeはDiscrete action spaceでは使えない)
    D = ExperienceReplay(args.experience_size, args.symbolic_env,
                         env.observation_size, env.num_action, args.bit_depth, args.device)
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(
            t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)


# Initialise model parameters randomly
#FIXME: env.action_sizeは使えない
transition_model = TransitionModel(belief_size=args.belief_size,
                                   state_size=args.state_size,
                                   num_action=env.num_action,
                                   hidden_size=args.hidden_size,
                                   embedding_size=args.embedding_size,
                                   activation_function=args.dense_activation_function,
                                   disable_gru_norm=args.disable_gru_norm,
                                   kl_free=args.kl_free,
                                   kl_scale=args.kl_scale,
                                   kl_balance=args.kl_balance
                                   ).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size,
                                     args.state_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
reward_model = RewardModel(h_size=args.belief_size, s_size=args.state_size, hidden_size=args.hidden_size,
                           activation=args.dense_activation_function).to(device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size,
                  args.embedding_size, args.cnn_activation_function).to(device=args.device)
actor_model = CategoricalActorModel(
                num_actions=env.num_action,
                h_size=args.belief_size,
                s_size=args.state_size,
                num_layers=args.num_actor_layers,
    num_units=args.num_actor_units
              ).to(device=args.device)
value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size,
                         args.dense_activation_function).to(device=args.device)
param_list = list(transition_model.parameters()) \
    + list(observation_model.parameters()) \
    + list(reward_model.parameters()) \
    + list(encoder.parameters())
value_actor_param_list = list(
    value_model.parameters()) \
    + list(actor_model.parameters())
params_list = param_list + value_actor_param_list
model_optimizer = optim.Adam(param_list, lr=0 if args.learning_rate_schedule !=
                             0 else args.model_learning_rate, eps=args.adam_epsilon)
actor_optimizer = optim.Adam(actor_model.parameters(
), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
value_optimizer = optim.Adam(value_model.parameters(
), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)
if args.models != '' and os.path.exists(args.models):
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])

# prepare schedulers
imag_grad_mix_sched = init_scheduler(config=args.imag_gradient_mix)
actor_state_entropy_sched = init_scheduler(config=args.actor_state_entropy)
actor_entropy_sched = init_scheduler(config=args.actor_entropy)

# prepare actor
if args.algo == "dreamer":
    print("DREAMER")
    planner = actor_model
else:
    #TODO: fix env.action_size(離散行動空間では使えない)
    planner = MPCPlanner(env.num_action, args.planning_horizon, args.optimisation_iters,
                         args.candidates, args.top_candidates, transition_model, reward_model)  # TODO: num_actionに変えたが大丈夫か確認
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(
    args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
# Allowed deviation in KL divergence
free_nats = torch.full((1, ), args.free_nats, device=args.device)


def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    belief, _, _, posterior_state, _ = transition_model(posterior_state, action.unsqueeze(
        dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0)  # Remove time dimension from belief/state
    if args.algo == "dreamer":
        action = planner.get_action(belief, posterior_state, det=not(explore))
    else:
        # Get action from planner(q(s_t|o≤t,a<t), p)
        action = planner(belief, posterior_state)
    if explore:
        # Add gaussian exploration noise on top of the sampled action
        action = torch.clamp(
            Normal(action, args.action_noise).rsample(), -1, 1)
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, done = env.step(action.cpu() if isinstance(
        env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            #FIXME: 離散行動空間ではenv.action_sizeは使えない
            belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(
                1, args.state_size, device=args.device), torch.zeros(1, env.num_action, device=args.device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(
                    args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
                total_reward += reward
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()


# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
    # Model fitting
    losses = []
    model_modules = transition_model.modules+encoder.modules + \
        observation_model.modules+reward_model.modules

    print("training loop")
    for s in tqdm(range(args.collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
            args.batch_size, args.chunk_size)  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(
            args.batch_size, args.state_size, device=args.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior_logits, posterior_states, posterior_logits = transition_model(
            init_state, actions[:-1], init_belief, bottle_tuple(encoder, (observations[1:], )), nonterminals[:-1])

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if args.worldmodel_LogProbLoss:
            observation_loss = - observation_model.get_log_prob(
                {'h_t': beliefs, 's_t': posterior_states, 'o_t': observations[1:]}, sum_features=False)
            observation_loss = observation_loss.sum(
                dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        else:
            observation_mean = observation_model(
                h_t=beliefs, s_t=posterior_states)['loc']
            observation_loss = F.mse_loss(observation_mean, observations[1:], reduction='none').sum(
                dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))

        if args.worldmodel_LogProbLoss:
            reward_loss = -reward_model.get_log_prob(
                {'h_t': beliefs, 's_t': posterior_states, 'r_t': rewards[:-1]}, sum_features=False)
            reward_loss = reward_loss.mean(dim=(0, 1))
        else:
            reward_mean = reward_model(
                h_t=beliefs, s_t=posterior_states)['loc']
            reward_loss = F.mse_loss(
                reward_mean, rewards[:-1], reduction='none').mean(dim=(0, 1))

        # transition loss
        kl_loss, value = transition_model.calc_kld(current_step=s,
                                            posterior_logits=posterior_logits,
                                            prior_logits=prior_logits) #TODO: valueの使いみちを著者実装で確認

        # Calculate latent overshooting objective for t > 0 (This corresponds to behavior learning)
        if args.overshooting_kl_beta != 0:
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, args.chunk_size - 1):
                d = min(t + args.overshooting_distance,
                        args.chunk_size - 1)  # Overshooting distance
                # Use t_ and d_ to deal with different time indexing for latent states
                t_, d_ = t - 1, d - 1
                # Calculate sequence padding so overshooting terms can be calculated in one batch
                seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior logits, (6) sequence masks
                overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_logits[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)))
                # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences

            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_logits = transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(
                overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
            seq_mask = torch.cat(overshooting_vars[6], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(Categorical(torch.cat(overshooting_vars[5], dim=1)), Categorical(
                prior_logits)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
            # Calculate overshooting reward prediction loss with sequence mask
            if args.overshooting_reward_scale != 0:
                reward_loss += (1 / args.overshooting_distance) * args.overshooting_reward_scale * F.mse_loss(reward_model(beliefs, prior_states)['loc'] * seq_mask[:, :, 0], torch.cat(
                    overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(group['lr'] + args.model_learning_rate /
                                  args.model_learning_rate_schedule, args.model_learning_rate)
        model_loss = observation_loss + reward_loss + kl_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
            actor_logits = posterior_logits.detach()

        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(
                actor_states, actor_beliefs, actor_logits, actor_model, transition_model, args.planning_horizon)
        imged_beliefs, imged_prior_states, imged_actions, imged_prior_logits = imagination_traj
        # ERASEME: 価値関数(value_moodel)は変わらず正規分布のままなので、そのまま使える
        with FreezeParameters(model_modules + value_model.modules):
            imged_reward = reward_model(
                h_t=imged_beliefs, s_t=imged_prior_states)['loc']
            value_pred = value_model(
                h_t=imged_beliefs, s_t=imged_prior_states)['loc']
        returns = lambda_return(imged_reward, value_pred,
                                bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
        actor_loss = -torch.mean(returns)
        reinforce_loss = actor_model.get_log_prob(
            {"a_t": imged_actions, "h_t": imged_beliefs, "s_t": imged_prior_states}, sum_features=False)[:-1]
        reinforce_loss *= returns.detach()[:-1] - value_pred[:-1]
        reinforce_loss = -torch.mean(reinforce_loss)

        ratio = imag_grad_mix_sched(s)
        actor_loss += ratio * actor_loss + (1 - ratio) * reinforce_loss

        # calculate actor entropy
        actor_entropy = actor_model.get_entropy({"h_t": imged_beliefs, "s_t": imged_prior_states}, sum_features=False)[:-1]
        actor_loss += torch.mean(actor_entropy) * actor_entropy_sched(s)

        # calculate transition_model's entropy
        actor_loss += torch.mean(Categorical(logits=imged_prior_logits).entropy()) * actor_state_entropy_sched(s)

        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(),
                                 args.grad_clip_norm, norm_type=2)
        actor_optimizer.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        # detach the input tensor from the transition network.
        value_loss = - value_model.get_log_prob(
            {'h_t': value_beliefs, 's_t': value_prior_states, 'r_t': target_return}, sum_features=False)
        value_loss = value_loss.mean(dim=(0, 1))
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(),
                                 args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append([observation_loss.item(), reward_loss.item(),
                       kl_loss.item(), actor_loss.item(), value_loss.item()])

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):],
             metrics['observation_loss'], 'observation_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):],
             metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):],
             metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']):],
             metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):],
             metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    print("Data collection")
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(
            1, args.state_size, device=args.device), torch.zeros(1, env.num_action, device=args.device)
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), explore=True)
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break

        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(metrics['episodes'][-len(metrics['train_rewards']):],
                 metrics['train_rewards'], 'train_rewards', results_dir)

    # Test model
    print("Test model")
    if episode % args.test_interval == 0:
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        encoder.eval()
        actor_model.eval()
        value_model.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed,
                                     args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(
            ), np.zeros((args.test_episodes, )), []
            #FIXME: 離散行動空間では使えない
            belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(
                args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                    args, test_envs, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
                total_rewards += reward.numpy()
                observation = next_observation
                if done.sum().item() == args.test_episodes:
                    pbar.close()
                    break

        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'],
                 'test_rewards', results_dir)
        lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
                 metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
        if not args.symbolic_env:
            episode_str = str(episode).zfill(len(str(args.episodes)))
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        actor_model.train()
        value_model.train()
        # Close test environments
        test_envs.close()

    writer.add_scalar(
        "train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward",
                      metrics['train_rewards'][-1], metrics['steps'][-1]*args.action_repeat)
    writer.add_scalar("observation_loss",
                      metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar(
        "reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar(
        "kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar(
        "actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar(
        "value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
    print("episodes: {}, total_steps: {}, train_reward: {} ".format(
        metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save({'transition_model': transition_model.state_dict(),
                    'observation_model': observation_model.state_dict(),
                    'reward_model': reward_model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'actor_model': actor_model.state_dict(),
                    'value_model': value_model.state_dict(),
                    'model_optimizer': model_optimizer.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict()
                    }, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            # Warning: will fail with MemoryError with large memory sizes
            torch.save(D, os.path.join(results_dir, 'experience.pth'))


# Close training environment
env.close()
