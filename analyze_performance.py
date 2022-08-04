import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cycler import cycler
import numpy as np
import statistics

import get_score


matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.linewidth'] = 3

def draw_throughput(x, y, y2, text_x, text_y, output, y2lim = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ln1 = ax.plot(x, y, zorder=3, label='Throughput', color='tab:green', ls='-')
    ax.set_title('Performance of AIDTN during\n different congestion levels', 
    color='grey', fontweight="bold")
    ax.grid(True, axis='y', linestyle='-', linewidth=1, which='major', 
    color='grey', alpha=0.5, zorder=0)
    ax.set_xlim([0, None])
    ax.set_ylim([0, text_y+5])    
    ax.set_xlabel("Elapsed time (s)", color='grey')
    ax.set_ylabel("Throughput (Gb/s)", color='grey')
    ax.axvspan(300, 600, facecolor='lightgreen', )
    ax.text(text_x, text_y, 'Congestion', color='black', zorder=6)
    ax.text(150, text_y, 'Before\nCongestion', color='black', 
    horizontalalignment='center', verticalalignment='center', zorder=5)
    ax.text(text_x+500, text_y, 'After\nCongestion', color='black', 
    horizontalalignment='center', verticalalignment='center', zorder=5)

    for k,v in ax.spines.items():
        v.set_color('tab:blue')
    ax.tick_params(axis="both", colors="grey")
    ax.legend(prop={'size': 15}, labelcolor='grey')
    plt.tight_layout()
    fig.savefig(output)

def draw_bargraph(dataset, indices,output, suptitle, 
metrices=['Goodput', 'Packet_losses'], 
barlabels = ['Underallocated', 'Optimized', 'Overallocated'],
ylimits=[[None,None],[None,None]], legend_locs=['best', 'best'], 
modifiers = [ 8/1e9, None], 
labels = ['Before\ncongestion', 'Congestion', 'After\ncongestion']):
    
    x = np.arange(len(labels))
    width = 0.2
    barwidth = 0.15
    fig, axes = plt.subplots(figsize=(10, 12), nrows=2)

    edgecolors = ['tab:blue', 'tab:red','tab:green' ,'tab:purple' ]
    hatches = ['||', '++', '--', '//']

    for i in range(len(dataset)):

        for k in range(len(metrices)):
            y = []
            for j in range(len(indices[0])-1):
                y.append(dataset[i][metrices[k]][indices[i][j] - 
                2:indices[i][j+1]-2].mean())
            #change B/s to Gbps
            if modifiers[k] != None : y = np.array(y) * modifiers[k]
            axes[k].bar(x - (width * ((len(dataset)-1)/2)) + (width*i), y, 
            barwidth, zorder=3, label=barlabels[i], color = 'white', 
            edgecolor=edgecolors[i], lw=1, hatch=hatches[i])

    for i in range(len(axes)):
        for k,v in axes[i].spines.items():
            v.set_color('tab:blue')
        axes[i].grid(True, axis='y', linestyle='-', linewidth=1, which='major', 
        color='grey', alpha=0.5, zorder=0)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, color='grey')
        if ylimits[i] != [None,None]: 
            axes[i].set_ylim(ylimits[i])
        axes[i].legend(prop={'size': 15}, labelcolor='grey', 
        loc=legend_locs[i])
    
    axes[0].tick_params(axis="y", colors="grey")
    axes[0].set_title('Throughput (Gb/s)', color='grey', fontweight="bold")
    axes[1].set_title("Packets lost (packets/s)", color='grey', 
    fontweight="bold")
    fig.suptitle(suptitle, color='grey', fontweight="bold")
    axes[0].patch.set_visible(False)

    plt.tight_layout()
    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.03, 0.46), 0.95, 0.425, fill=False, color="k", lw=2, 
        zorder=1000, transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.03, 0.02), 0.95, 0.44, fill=False, color="k", lw=2, 
        zorder=1000, transform=fig.transFigure, figure=fig)

    fig.patches.extend([rect])
    #plt.show()
    fig.savefig(output)
    plt.close()

def draw_scatter(dataset, output, title, 
barlabels = ['Underallocated', 'Optimized', 'Overallocated'], 
ax1_ylimit=[None,None], legend1_loc='best', arrow=None):

    colors = ['tab:blue', 'tab:red','tab:green' ]
    fmts = ['-o', '-^', '-D']
    fig, ax = plt.subplots(figsize=(10, 6))
    for k,v in ax.spines.items():
        v.set_color('tab:blue')            

    for i in range(len(dataset)):
        thr = statistics.mean(dataset[i].iloc[0]) * 8/1e9
        std = statistics.stdev(dataset[i].iloc[0])* 8/1e9
        pktloss = statistics.mean(dataset[i].iloc[1]) 
        pkl_std = statistics.stdev(dataset[i].iloc[1])
        ax.errorbar( pktloss, thr, yerr=std, xerr=pkl_std, zorder=3, 
        label=barlabels[0], color=colors[i], lw=2, fmt=fmts[i], capsize=3, 
        markersize=20)    
    
    if arrow != None: ax.add_patch(arrow)
    ax.grid(True, axis='both', linestyle='-', linewidth=1, which='major', color='grey', alpha=0.5, zorder=0)
    ax.tick_params(axis="both", colors="grey")    
    ax.set_ylim(ax1_ylimit)
    ax.set_xlabel("Packet losses (packets)", color='grey', fontweight="bold")
    ax.set_ylabel("Throughput (Gb/s)", color='grey', fontweight="bold")
    ax.set_title(title, color='grey', fontweight="bold")
    ax.legend(prop={'size': 15}, labelcolor='grey', loc=legend1_loc)

    ax.scatter(0, 35.5, s=1500, marker="*" )
    ax.text(2500, 34, "More efficient", rotation=-15)
    ax.text(-500, 36, "   Most\n efficient")

    ax.patch.set_visible(False)
    plt.tight_layout()
    #plt.show()
    fig.savefig(output)
    plt.close()

print('Training Prediction model with PRP dataset')
get_score.get_train_throughput_rmse_prp()
print('Training Prediction model with MRP dataset')
get_score.get_train_throughput_rmse_mrp()

# PRP data
print('Generaing AIDTN performance using PRP dataset')
dynamic = pandas.read_csv('dataset/prp/dynamic.csv', parse_dates=True, 
index_col=0)
underalloc = pandas.read_csv('dataset/prp/underalloc.csv', parse_dates=True, 
index_col=0)
overalloc = pandas.read_csv('dataset/prp/overalloc.csv', parse_dates=True, 
index_col=0)

# start, scaleup, scale down, end indexes found from the dataset
underalloc_times = [3, 37, 56, 159]
dynamic_times = [3, 23, 50, 76]
overalloc_times = [3, 27, 46, 83]

draw_throughput((dynamic.index - dynamic.index[0]).seconds, 
dynamic['Goodput']*8/1e9, dynamic['Packet_losses'],  320, 35, 
'results/congestion.pdf', y2lim = 4000)
draw_bargraph([underalloc, dynamic, overalloc], 
[underalloc_times, dynamic_times, overalloc_times], 'results/prp.pdf', 
"Comparison between schemes \nwith AIDTN in PRP")

# NVMeoF data
print('Done\nGeneraing AIDTN NVMeoF performance using PRP dataset')
underalloc = pandas.read_csv('dataset/mrp/underalloc.csv', parse_dates=True, 
index_col=0)
dynamic = pandas.read_csv('dataset/mrp/dynamic.csv', parse_dates=True, 
index_col=0)
overalloc = pandas.read_csv('dataset/mrp/overalloc.csv', parse_dates=True, 
index_col=0)

underalloc_times = [3, 31, 38, 204]
dynamic_times = [5, 24, 30, 104]
overalloc_times = [3, 54, 61, 102]

draw_bargraph([underalloc, dynamic, overalloc], 
[underalloc_times, dynamic_times, overalloc_times], 'results/nvmeof_mrp.pdf', 
"Comparison between schemes with AIDTN \nusing NVMeoF transport", 
metrices=['NVMe_from_transfer', 'Packet_losses'])


print('Done\nGeneraing AIDTN Feature comparision using MRP dataset')
net = pandas.read_csv('dataset/Feature_comparison/net.csv', header=None)
sys_net = pandas.read_csv('dataset/Feature_comparison/sys_net.csv', header=None)
sys_net_nvme = pandas.read_csv('dataset/Feature_comparison/sys_net_nvme.csv', 
header=None)

arrow = mpatches.FancyArrowPatch((6500, 33), (300, 35.4),
                                 mutation_scale=30, arrowstyle='-|>')

draw_scatter([net, sys_net, sys_net_nvme],  
'results/features_comparison.pdf', 
"Comparision of AIDTN\n with different features",
barlabels=['net', 'net+sys', 'net+sys+nvme'], ax1_ylimit=[None, 35.9], 
arrow=arrow)

print('Done\nGeneraing AI algorithm comparision using MRP dataset')

AIDTN = pandas.read_csv('dataset/AI_comparison/AIDTN.csv', parse_dates=True, 
index_col=0)
PCP = pandas.read_csv('dataset/AI_comparison/PCP.csv', parse_dates=True, 
index_col=0)
Ernest = pandas.read_csv('dataset/AI_comparison/Earnest.csv', parse_dates=True,
index_col=0)
AIDTNnet = pandas.read_csv('dataset/AI_comparison/netAI.csv', parse_dates=True,
index_col=0)

# start, scaleup, scale down
pcp_i = [3, 29, 37]
AIDTN_i = [3, 18, 25]
Ernest_i = [3, 29, 37]
ainet_i = [16, 27, 35]

draw_bargraph([PCP, AIDTN, Ernest, AIDTNnet], 
[pcp_i, AIDTN_i, Ernest_i, ainet_i], 'results/AI_comparison.pdf', 
"Comparison between AIDTN and \nother optimization algorithms",
barlabels = ['PCP+', 'AIDTN', 'Ernest', 'netAI'],
labels = ['Without Congestion', 'With Congestion'])
print('Done')