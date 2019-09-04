import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os.path import isdir, join
from tqdm import tqdm
from os import makedirs
import shutil
import mocap.visualization.human_pose as hviz

# #318BA3
# #14ABBD
# #FFAA00
# #FF6000
# #FF3800


class SequenceVisualizer:

    def __init__(self, data_root, name,
                 plot_fn=None,
                 vmin=-1, vmax=1,
                 to_file=False, subsampling=1,
                 with_pauses=False,
                 fps=20,
                 mark_origin=False):
        """
        :param data_root:
        :param name:
        :param seq:
        :param vmin:
        :param vmax:
        :param fps:
        :param to_file: if True write to file
        :param subsampling
        """
        assert isdir(data_root)
        self.name = name
        self.with_pauses = with_pauses
        seq_root = join(data_root, name)
        self.seq_root = seq_root
        self.mark_origin = mark_origin
        self.fps = fps
        self.plot_fn = plot_fn

        if to_file and isdir(seq_root):
            print("[visualizer] delete ", seq_root)
            shutil.rmtree(seq_root)

        if to_file:
            print('[visualizer] write to ', seq_root)

        self.vmin = vmin
        self.vmax = vmax
        self.subsampling = subsampling
        self.to_file = to_file
        self.counter = 0

    def plot(self, seq1, seq2=None, parallel=False,
             plot_fn1=None, plot_fn2=None,
             parallel_distance=0,
             views=[  # (elevation, azimuth)
                 (45., 45.)
             ],
             lcolor2="#E1C200", rcolor2='#5FBF43',
             lcolor='#099487', rcolor='#F51836',
             noaxis=False, noclear=False,
             toggle_color=False,
             plot_cbc=None, last_frame=None,
             definite_cbc=None,
             name='',
             plot_jid=False):
        """
        # 002540
        # 099487
        # 5FBF43
        # E1C200
        # F51836
        :param seq:
        :param seq2: is being plotted after seq
        :param parallel: if True, plot seq2 parallel to seq
        :param plot_fn1: plot function for seq1
        :param plot_fn2: plot function for seq2
        :param lcolor:
        :param rcolor:
        :param lcolor2:
        :param rcolor2:
        :param noaxis:
        :param plot_cbc: def plot_cvc(ax, seq, t):
        :param definite_cbc: def plot_cbc(ax, i, t): will be called for sure
        :param last_frame:
        :param name: string added to the newly created folder name
        :param plot_jid:
        :return:
        """
        if last_frame is None:
            if seq2 is None or parallel:
                last_frame = len(seq1)
                n = last_frame
                if parallel:
                    assert seq2 is not None
                    assert len(seq2) == last_frame
            else:
                n = len(seq1)
                seq1 = np.concatenate([seq1, seq2], axis=0)
                last_frame = len(seq1)

        if toggle_color:
            assert not parallel
            assert seq2 is None
        
        if plot_fn1 is None:
            plot_fn1 = self.plot_fn
        if plot_fn2 is None:
            plot_fn2 = plot_fn1

        counter = self.counter
        self.counter += 1  # so we can plot multiple videos!
        vmin = self.vmin
        vmax = self.vmax
        mark_origin = self.mark_origin
        subsampling = self.subsampling
        to_file = self.to_file
        seq_root = self.seq_root
        video_dir = join(seq_root, 'seq' + str(counter) + name)
        if to_file:
            assert not isdir(video_dir)
            makedirs(video_dir)

        n_views = len(views)
        fig = plt.figure(figsize=(n_views * 9, 9))

        Axs = []
        for vv in range(n_views):
            ax = fig.add_subplot(1, n_views, vv+1, projection='3d')
            Axs.append(ax)

        for iii, t in enumerate(tqdm(range(0, last_frame, subsampling))):
            for vv, (elev, azim) in enumerate(views):
                ax = Axs[vv]
                if not noclear:
                    ax.clear()
                if noaxis:
                    ax.axis('off')
                ax.set_xlim([vmin, vmax])
                ax.set_ylim([vmin, vmax])
                ax.set_zlim([vmin, vmax])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                if mark_origin:
                    ax.plot([0, 0], [-10, 10], [0, 0], color='black', alpha=0.5)
                    ax.plot([-10, 10], [0, 0], [0, 0], color='black', alpha=0.5)
                if plot_cbc is None:
                    if parallel:
                        T1 = np.array([parallel_distance/2, 0, 0])
                        T2 = np.array([-parallel_distance/2, 0, 0])
                        if plot_fn1 is None:
                            hviz.plot(ax, seq1[t], lcolor=lcolor, rcolor=rcolor, plot_jid=plot_jid)
                        else:
                            plot_fn1(ax, seq1[t], lcolor=lcolor, rcolor=rcolor, plot_jid=plot_jid, T=T1)
                        if plot_fn2 is None:
                            hviz.plot(ax, seq2[t], lcolor=lcolor2, rcolor=rcolor2, plot_jid=plot_jid)
                        else:
                            plot_fn2(ax, seq2[t], lcolor=lcolor2, rcolor=rcolor2, plot_jid=plot_jid, T=T2)

                    elif toggle_color:
                        _lcolor = lcolor if t % 2 == 0 else lcolor2
                        _rcolor = rcolor if t % 2 == 0 else rcolor2
                        if plot_fn1 is None:
                            hviz.plot(ax, seq1[t], lcolor=_lcolor, rcolor=_rcolor, plot_jid=plot_jid)
                        else:
                            plot_fn1(ax, seq1[t], lcolor=_lcolor, rcolor=_rcolor, plot_jid=plot_jid)
                    else:  # temporal
                        if t < n:
                            if plot_fn1 is None:
                                hviz.plot(ax, seq1[t], lcolor=lcolor, rcolor=rcolor, plot_jid=plot_jid)
                            else:
                                plot_fn1(ax, seq1[t], lcolor=lcolor, rcolor=rcolor, plot_jid=plot_jid)
                        else:
                            if plot_fn2 is None:
                                hviz.plot(ax, seq1[t], lcolor=lcolor2, rcolor=rcolor2, plot_jid=plot_jid)
                            else:
                                plot_fn2(ax, seq1[t], lcolor=lcolor2, rcolor=rcolor2, plot_jid=plot_jid)
                else:
                    plot_cbc(ax, seq1, t)

                ax.set_title('frame ' + str(t + 1))
                
                if definite_cbc is not None:
                    definite_cbc(ax, iii, t)

            if to_file:
                if n_views > 1:
                    fig.savefig(join(video_dir, 'out%05d.png' % t), 
                    pad_inches=0,
                    transparent=False)
                else:
                    extent = ax.get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted())
                    fig.savefig(join(video_dir, 'out%05d.png' % t),
                                bbox_inches=extent,
                                pad_inches=0,
                                transparent=False)
            else:
                plt.pause(1 / self.fps)
                if self.with_pauses:
                    plt.waitforbuttonpress()

        if not to_file:
            plt.show()
            plt.close()
