#! /usr/bin/env python

# Project:  Lihang
# Filename: utils
# Date: 6/12/19
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import matplotlib.pyplot as plt


class Radar(object):
    def __init__(self, feas=None, labels=None):
        self.feas = feas
        self.colors = ['green', 'blue', 'red', 'yellow', 'black']
        self.labels = labels

    def plot(self, data):
        assert self.feas is not None

        angles = np.linspace(0.1 * np.pi, 2.1 * np.pi, len(self.feas),
                             endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)

        # ax.legend(loc=[0.25, 1.15], fontsize=18)
        ax.set_yticklabels([])
        ax.set_thetagrids(angles * 180/np.pi, self.feas, fontsize=12)
        ax.grid(True)

        for idx, label in enumerate(self.labels):
            stats = data[idx]
            stats = np.concatenate((stats, [stats[0]]))
            ax.plot(angles, stats, '--', linewidth=1, 
                    c=self.colors[idx % len(self.colors)],
                    label=str(label))
            ax.fill(angles, stats, 
                    c=self.colors[idx % len(self.colors)],
                    alpha=0.25)
        plt.legend()
        plt.show()
        return fig
