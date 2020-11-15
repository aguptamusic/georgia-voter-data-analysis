import numpy as np
import copy
from voting_utils import get_filepath
import matplotlib.pyplot as plt

"""
Name: Avi Gupta
Email: agupta07@stanford.edu
"""

"""
Create each of four stacked bar plots representing
percent of voters of a given race that voted in each 
county in 2018 and 2020.
"""
def plot_voter_turnout(pct_18, pct_diff, race):
    num_counties = len(pct_18)  # 159 counties
    x_loc = np.arange(0, num_counties, 1)  # x locations for counties

    w_plot_18 = plt.bar(x_loc, pct_18)
    w_plot_diff = plt.bar(x_loc, pct_diff, bottom=pct_18)
    plt.xlabel('County')
    plt.ylabel('% Voted')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.title('% of {} People that Voted by County'.format(race))
    plt.legend((w_plot_18[0], w_plot_diff[0]), ('2018', '2020'))
    plt.show()

"""
Read in data and call plotting helper function. 
"""
def graph_percentages(voters_18, voters_20):
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

    # difference
    w_pct_diff = w_pct_20 - w_pct_18
    b_pct_diff = b_pct_20 - b_pct_18
    h_pct_diff = h_pct_20 - h_pct_18
    o_pct_diff = o_pct_20 - o_pct_18

    # plot
    plot_voter_turnout(w_pct_18, w_pct_diff, "White")
    plot_voter_turnout(b_pct_18, b_pct_diff, "Black")
    plot_voter_turnout(h_pct_18, h_pct_diff, "Hispanic/Latinx")
    plot_voter_turnout(o_pct_18, o_pct_diff, "Other")


def main():
    voters_18 = get_filepath('2018_Voters.csv')
    voters_20 = get_filepath('2020_Voters.csv')
    graph_percentages(voters_18, voters_20)

if __name__ == "__main__":
    main()
