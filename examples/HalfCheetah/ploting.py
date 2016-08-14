import logging
logging.disable(logging.CRITICAL)
import sys
sys.dont_write_bytecode = True
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robustRL.algos import *
from robustRL.samplers import *

np.random.seed(10)
rllab_set_seed(10)

def visulaise_agent(job_id=0,
    hidden_sizes = (32,32),
    max_kl = 0.01,
    gamma = 0.995,
    num_cpu = 'max',
    env_mode = 'test',
    normalized_env = False,
    pol_restart_file = None):

    
    e = get_environment(env_mode)
    #policy = GaussianMLPPolicy(env_spec=e.spec, hidden_sizes=hidden_sizes)
    #baseline = LinearFeatureBaseline(env_spec=e.spec)
    
    #Its a strong assumption that policy restart files is provided
    if pol_restart_file != None:
        policy = pickle.load(open(pol_restart_file, 'rb'))

    param_values = [2, 3, 4, 5, 6.3, 7, 8, 9, 10, 11, 12, 13, 14]
    niter = 100
    mean_stats = np.zeros(len(param_values))
    percentile_15_stats = np.zeros(len(param_values))
    percentile_85_stats = np.zeros(len(param_values))
    #train_curve = np.array(len(param_values))
    for i, param in enumerate(param_values):
        print param
        #temp_series = policy_evaluation(policy, 'test', num_episodes=niter, visual=False, param=param)
        temp_parallel = policy_evaluation_parallel(policy, 'test', num_episodes=niter, visual=False, param=param)
        e = get_environment(env_mode, param=param)
        base, perc, _ = e.evaluate_policy(policy, num_episodes=niter, percentile=[15, 85])
        print "def: ",base[0]
        #print "series: ", temp_series[0]
        print "parallel: ", temp_parallel[0]
        mean_stats[i] = base[0]
        percentile_15_stats[i] = perc[0]
        percentile_85_stats[i] = perc[1]
        
        #train_curve[i] = stats[0]
    
        
    #plot the graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(param_values, mean_stats, lw=2, label='mean', color='blue')
    
    ax.fill_between(param_values, percentile_85_stats, percentile_15_stats, facecolor='green', alpha=0.3, label='15-85 percentile')
    ax.legend(loc=4)
    plt.xlabel('Masses')
    plt.ylabel('Avg returns')
    ax.annotate('(%s, %s)' % (param_values[3], mean_stats[3]), xy=(param_values[3], mean_stats[3]), textcoords='data')
    plt.savefig('Results_for_various_mass_values')  
    plt.close()
    
if __name__ == '__main__':
    visulaise_agent(pol_restart_file='/home/veer/summer16/robustRL/examples/HalfCheetah/experiment_0/iterations/policy_431.pickle')
