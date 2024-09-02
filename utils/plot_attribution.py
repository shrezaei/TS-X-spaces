#based on TSInterpret Implementation

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.utils import create_path_if_not_exists


def plot_attribution(item, exp, figsize=(6.4, 4.8), normalize_attribution=True, save_path=None, title=""):
    """
    Plots explanation on the explained Sample.

    Arguments:
        item np.array: instance to be explained,if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`.
        exp np.array: explanation, ,if `mode = time`->`(time,feat)`  or `mode = feat`->`(feat,time)`.
        figsize (int,int): desired size of plot.
        heatmap bool: 'True' if only heatmap, otherwise 'False'.
        save str: Path to save figure.
    """
    if len(item[0]) == 1:
        test = item[0]
        # if only one-dimensional input
        fig, (axn, cbar_ax) = plt.subplots(
            len(item[0]), 2, sharex=False, sharey=False, figsize=figsize, gridspec_kw={'width_ratios': [40, 1]},
        )

        # Shahbaz: Set color pallete such that negative is red and positive is blue
        my_cmap = sns.diverging_palette(260, 10, as_cmap=True)

        #set min and max to have same absolute values
        extremum = np.max(abs(exp))

        # cbar_ax = fig.add_axes([.91, .3, .03, .4])
        axn012 = axn.twinx()
        if normalize_attribution:
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap=my_cmap,
                ax=axn,
                yticklabels=False,
                vmin=-1*extremum,
                vmax=extremum,
                cbar_ax=cbar_ax,
                # cbar_kws={"orientation": "vertical"},
            )
        else:
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                ax=axn,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
        sns.lineplot(
            x=np.arange(0, len(item[0][0].reshape(-1))) + 0.5,
            y=item[0][0].flatten(),
            ax=axn012,
            color="black",
        )
        # plt.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.95, top=0.95, bottom=0.05)
        cbar_ax.tick_params(labelsize=10)
        plt.title(title)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    else:
        ax011 = []

        fig, axn = plt.subplots(
            len(item[0]), 1, sharex=True, sharey=True, figsize=figsize
        )
        cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

        for channel in item[0]:
            # print(item.shape)
            # ax011.append(plt.subplot(len(item[0]),1,i+1))
            # ax012.append(ax011[i].twinx())
            # ax011[i].set_facecolor("#440154FF")
            axn012 = axn[i].twinx()
            if normalize_attribution:

                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=np.min(exp),
                    vmax=np.max(exp),
                )
            else:
                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=0,
                    vmax=1,
                )

            sns.lineplot(
                x=range(0, len(channel.reshape(-1))),
                y=channel.flatten(),
                ax=axn012,
                color="white",
            )
            plt.xlabel("Time", fontweight="bold", fontsize="large")
            plt.ylabel(f"Feature {i}", fontweight="bold", fontsize="large")
            i = i + 1
        fig.tight_layout(rect=[0, 0, 0.9, 1])
    if save_path is not None:
        path_only = save_path.rsplit("/", 1)[0] + "/"
        create_path_if_not_exists(path_only)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
