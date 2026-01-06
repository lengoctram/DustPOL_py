from numpy import *
from pylab import *
from matplotlib import *
import matplotlib.pyplot as plt
from itertools import cycle
from collections import OrderedDict

lsN                  = OrderedDict(
                                 [
                                  ('solid',               (0, ())),
                                  ('dashed',              (0, (5, 5))),
                                  # ('solid',               (0, ())),
                                  ('dashdotted',          (0, (5, 4, 1, 6))),
                                  ('dotted',              (0, (1, 5))),

                                  ('loosely dashed',      (0, (5, 15))),
                                  ('densely dashed',      (0, (5, 1))),
                                  ('loosely dotted',      (0, (1, 10))),
                                  ('densely dotted',      (0, (1, 1))), 
                                  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

                                  ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                                  ('densely dashdotted',  (0, (3, 1, 1, 1))),

                                  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
ls = list(lsN.keys())

# configuration of plot
lncycler = cycle(['solid','dashed','dashdot',lsN['dashdotdotted'],'dotted',lsN['loosely dashed']])
symbcycler = cycle(['*','o','s','X','D','P','^'])
clrcycler = cycle(['black','green','blue','red','deepskyblue','magenta','darkorange'])

rcParams.update({
	'font.family': 'sans-serif',
	'mathtext.fontset' : 'stixsans',
	'font.size' : 17,
	'lines.linewidth' : 1.6,
	'legend.fontsize' : 15,
	'axes.labelsize' : 17,
	'font.weight': 'light'
})

props = dict(boxstyle='round',facecolor='white',alpha=0.5)

def custm_axis(ax,Ax1,Ax2,Ay1,Ay2):
	ax.xaxis.set_label_coords(Ax1,Ax2)
	ax.yaxis.set_label_coords(Ay1,Ay2)
	ax.xaxis.tick_bottom()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.tick_left()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.tick_params(which='major', direction='in', length=8, width=2, labelsize=15)
	ax.tick_params(which='minor', direction='in', length=4, width=1, labelsize=10)
	return

def custm_legend(leg):
	for line,text in zip(leg.get_lines(), leg.get_texts()):
		text.set_color(line.get_color())
	return
