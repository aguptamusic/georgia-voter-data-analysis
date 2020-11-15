import numpy as np
import copy
from voting_utils import get_filepath
import matplotlib.pyplot as plt

"""
Name: Avi Gupta
Email: agupta07@stanford.edu
"""

"""
Calculate weighted average of voting probabilities for a specific
race/ethnicity across all counties in Georgia. Weighted by
proportion of population residing in the given county in that year. 
"""
def w_avg(pct_array, data, race):
    # get population array for given race and sum all populations
    pop = []
    if race == "White":
        pop = np.genfromtxt(data, delimiter=',', usecols=1, skip_header=1)
    elif race == "Black":
        pop = np.genfromtxt(data, delimiter=',', usecols=4, skip_header=1)
    elif race == "Hispanic":
        pop = np.genfromtxt(data, delimiter=',', usecols=7, skip_header=1)
    elif race == "Other":
        pop = np.genfromtxt(data, delimiter=',', usecols=10, skip_header=1)
    pop_sum = np.sum(pop)

    # calculate each weight and multiply by each probability
    avg = 0
    for i in range(len(pop)):
        weight = pop[i]/pop_sum
        avg += weight * pct_array[i]
    return avg

"""
 For a given race/ethnicity, assume null hypothesis that 
 the two voting samples were drawn from the same year. 
 Use bootstrapping to estimate p-value for difference 
 in the weighted averages of voter turnout
 probabilities across all counties between 2018 and 2020. 
"""
def bootstrap_p_val(pct_array_18, pct_array_20, voters_18, voters_20, race):
    obs_diff = abs(w_avg(pct_array_20, voters_20, race) - w_avg(pct_array_18, voters_18, race))
    print("Observed Difference in Weighted Avg. for {}: ".format(race), obs_diff)
    count = 0
    universal_sample = pct_array_18 + pct_array_20
    niters = 10000
    for i in range(niters):
        resample1 = np.random.choice(universal_sample, len(pct_array_18), replace=True)
        resample2 = np.random.choice(universal_sample, len(pct_array_20), replace=True)
        resample_diff = abs(w_avg(resample2, voters_20, race) - w_avg(resample1, voters_18, race))
        if resample_diff >= obs_diff:
            count += 1
    print("Bootstrapped p-value for {}: ".format(race), count/niters)

"""
Read in data and call bootstrapping function for each race group. 
"""
def get_p_val(voters_18, voters_20):
    # 2018
    w_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=2, skip_header=1)
    b_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=5, skip_header=1)
    h_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=8, skip_header=1)
    o_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=11, skip_header=1)

    # 2020
    w_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=2, skip_header=1)
    b_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=5, skip_header=1)
    h_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=8, skip_header=1)
    o_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=11, skip_header=1)

    # bootstrap p-val for difference in means for each race
    bootstrap_p_val(w_pct_18, w_pct_20, voters_18, voters_20, "White")
    bootstrap_p_val(b_pct_18, b_pct_20, voters_18, voters_20, "Black")
    bootstrap_p_val(h_pct_18, h_pct_20, voters_18, voters_20,"Hispanic")
    bootstrap_p_val(o_pct_18, o_pct_20, voters_18, voters_20, "Other")

"""
Plot average turnout across all counties per race groups in both years. 
"""
def plot_turnout(voters_18, voters_20):
    race_labels = ["White", "Black", "Hispanic", "Other"]
    # 2018
    w_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=2, skip_header=1)
    b_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=5, skip_header=1)
    h_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=8, skip_header=1)
    o_pct_18 = np.genfromtxt(voters_18, delimiter=',', usecols=11, skip_header=1)
    avg_white_18 = w_avg(w_pct_18, voters_18, race_labels[0])
    avg_black_18 = w_avg(b_pct_18, voters_18, race_labels[1])
    avg_hispanic_18 = w_avg(h_pct_18, voters_18, race_labels[2])
    avg_other_18 = w_avg(o_pct_18, voters_18, race_labels[3])

    # 2020
    w_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=2, skip_header=1)
    b_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=5, skip_header=1)
    h_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=8, skip_header=1)
    o_pct_20 = np.genfromtxt(voters_20, delimiter=',', usecols=11, skip_header=1)
    avg_white_20 = w_avg(w_pct_20, voters_20, race_labels[0])
    avg_black_20 = w_avg(b_pct_20, voters_20, race_labels[1])
    avg_hispanic_20 = w_avg(h_pct_20, voters_20, race_labels[2])
    avg_other_20 = w_avg(o_pct_20, voters_20, race_labels[3])

    # std. dev error bars
    n = len(w_pct_18)
    stderr_18 = [np.std(w_pct_18), np.std(b_pct_18), np.std(h_pct_18), np.std(o_pct_18)]
    stderr_20 = [np.std(w_pct_20), np.std(b_pct_20), np.std(h_pct_20), np.std(o_pct_20)]

    # create graph
    turnout_20 = [avg_white_20, avg_black_20, avg_hispanic_20, avg_other_20]
    turnout_18 = [avg_white_18, avg_black_18, avg_hispanic_18, avg_other_18]
    width = 0.35
    ind = np.arange(len(race_labels))
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, turnout_18, width, label='2018', yerr=stderr_18)
    rects2 = ax.bar(ind + width / 2, turnout_20, width, label='2020', yerr=stderr_20)

    ax.set_ylabel('% Voted')
    ax.set_xticks(ind)
    ax.set_title('Average Voter Turnout by Race')
    ax.set_xticklabels(race_labels)
    ax.legend()

    # label
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()

def main():
    voters_18 = get_filepath('2018_Voters.csv')
    voters_20 = get_filepath('2020_Voters.csv')
    plot_turnout(voters_18, voters_20)
    get_p_val(voters_18, voters_20)

if __name__ == "__main__":
    main()