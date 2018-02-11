import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import BoundaryNorm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cbook as cbook

from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.dates as mdates
import subprocess
import sys
import re
import datetime


def plotTest001():
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("../data/plot/matplotlib-pyplot-plot.png")
    plt.show()


def plotTest002():
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)

    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('A tale of 2 subplots')
    plt.ylabel('Damped oscillation')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')

    plt.show()


def plotTest003001():
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1  # difference of Gaussians

    im = plt.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
                    origin='lower', extent=[-3, 3, -3, 3],
                    vmax=abs(Z).max(), vmin=-abs(Z).max())

    plt.show()


def plotTest003002():
    # A sample image
    with cbook.get_sample_data('ada.png') as image_file:
        image = plt.imread(image_file)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')  # clear x- and y-axes

    # And another image

    w, h = 512, 512

    with cbook.get_sample_data('ct.raw.gz', asfileobj=True) as datafile:
        s = datafile.read()
    A = np.fromstring(s, np.uint16).astype(float).reshape((w, h))
    A /= A.max()

    fig, ax = plt.subplots()
    extent = (0, 25, 0, 25)
    im = ax.imshow(A, cmap=plt.cm.hot, origin='upper', extent=extent)

    markers = [(15.9, 14.5), (16.8, 15)]
    x, y = zip(*markers)
    ax.plot(x, y, 'o')

    ax.set_title('CT density')

    plt.show()


def plotTest003003():
    x = np.arange(120).reshape((10, 12))

    interp = 'bilinear'
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))
    axs[0].set_title('blue should be up')
    axs[0].imshow(x, origin='upper', interpolation=interp)

    axs[1].set_title('blue should be down')
    axs[1].imshow(x, origin='lower', interpolation=interp)

    plt.show()


def plotTest003004():
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1  # difference of Gaussians

    path = Path([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]])
    patch = PathPatch(path, facecolor='none')

    fig, ax = plt.subplots()
    ax.add_patch(patch)

    im = ax.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                   origin='lower', extent=[-3, 3, -3, 3],
                   clip_path=patch, clip_on=True)
    im.set_clip_path(patch)
    plt.show()


def plotTest004():
    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]

    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    ax0.set_title('pcolormesh with levels')

    # contours are *point* based plots, so convert our bound into point
    # centers
    cf = ax1.contourf(x[:-1, :-1] + dx / 2.,
                      y[:-1, :-1] + dy / 2., z, levels=levels,
                      cmap=cmap)
    fig.colorbar(cf, ax=ax1)
    ax1.set_title('contourf with levels')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    plt.show()


def plotTest005():
    np.random.seed(19680801)

    # example data
    mu = 100  # mean of distribution
    sigma = 15  # standard deviation of distribution
    x = mu + sigma * np.random.randn(437)

    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, '--')
    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def plotTest006():
    fig, ax = plt.subplots()

    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (1.58, -2.57)),
        (Path.CURVE4, (0.35, -1.1)),
        (Path.CURVE4, (-1.75, 2.0)),
        (Path.CURVE4, (0.375, 2.0)),
        (Path.LINETO, (0.85, 1.15)),
        (Path.CURVE4, (2.2, 3.2)),
        (Path.CURVE4, (3, 0.05)),
        (Path.CURVE4, (2.0, -0.5)),
        (Path.CLOSEPOLY, (1.58, -2.57)),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='r', alpha=0.5)
    ax.add_patch(patch)

    # plot control points and connecting lines
    x, y = zip(*path.vertices)
    line, = ax.plot(x, y, 'go-')

    ax.grid()
    ax.axis('equal')
    plt.show()


def plotTest007():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plotTest008():
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2
    speed = np.sqrt(U * U + V * V)

    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Varying Density')

    # Varying color along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
    fig.colorbar(strm.lines)
    ax1.set_title('Varying Color')

    #  Varying line width along a streamline
    ax2 = fig.add_subplot(gs[1, 0])
    lw = 5 * speed / speed.max()
    ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)
    ax2.set_title('Varying Line Width')

    # Controlling the starting points of the streamlines
    seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1, 0, 1, 2, 2]])

    ax3 = fig.add_subplot(gs[1, 1])
    strm = ax3.streamplot(X, Y, U, V, color=U, linewidth=2,
                          cmap='autumn', start_points=seed_points.T)
    fig.colorbar(strm.lines)
    ax3.set_title('Controlling Starting Points')

    # Displaying the starting points with blue symbols.
    ax3.plot(seed_points[0], seed_points[1], 'bo')
    ax3.axis((-w, w, -w, w))

    # Create a mask
    mask = np.zeros(U.shape, dtype=bool)
    mask[40:60, 40:60] = True
    U[:20, :20] = np.nan
    U = np.ma.array(U, mask=mask)

    ax4 = fig.add_subplot(gs[2:, :])
    ax4.streamplot(X, Y, U, V, color='r')
    ax4.set_title('Streamplot with Masking')

    ax4.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5,
               interpolation='nearest', cmap='gray', aspect='auto')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plotTest009():
    NUM = 250

    ells = [Ellipse(xy=np.random.rand(2) * 10,
                    width=np.random.rand(), height=np.random.rand(),
                    angle=np.random.rand() * 360)
            for i in range(NUM)]

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(np.random.rand())
        e.set_facecolor(np.random.rand(3))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.show()


def plotTest010():
    n_groups = 5

    means_men = (20, 35, 30, 35, 27)
    std_men = (2, 3, 4, 1, 2)

    means_women = (25, 32, 34, 20, 25)
    std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_men, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_men, error_kw=error_config,
                    label='Men')

    rects2 = ax.bar(index + bar_width, means_women, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_women, error_kw=error_config,
                    label='Women')

    ax.set_xlabel('Group')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
    ax.legend()

    fig.tight_layout()
    plt.show()


Student = namedtuple('Student', ['name', 'grade', 'gender'])
Score = namedtuple('Score', ['score', 'percentile'])
# GLOBAL CONSTANTS
testNames = ['Pacer Test', 'Flexed Arm\n Hang', 'Mile Run', 'Agility', 'Push Ups']
testMeta = dict(zip(testNames, ['laps', 'sec', 'min:sec', 'sec', '']))


def attach_ordinal(num):
    """helper function to add ordinal string to integers

    1 -> 1st
    56 -> 56th
    """
    suffixes = dict((str(i), v) for i, v in
                    enumerate(['th', 'st', 'nd', 'rd', 'th',
                               'th', 'th', 'th', 'th', 'th']))

    v = str(num)
    # special case early teens
    if v in {'11', '12', '13'}:
        return v + 'th'
    return v + suffixes[v[-1]]


def format_score(scr, test):
    """
    Build up the score labels for the right Y-axis by first
    appending a carriage return to each string and then tacking on
    the appropriate meta information (i.e., 'laps' vs 'seconds'). We
    want the labels centered on the ticks, so if there is no meta
    info (like for pushups) then don't add the carriage return to
    the string
    """
    md = testMeta[test]
    if md:
        return '{0}\n{1}'.format(scr, md)
    else:
        return scr


def format_ycursor(y):
    y = int(y)
    if y < 0 or y >= len(testNames):
        return ''
    else:
        return testNames[y]


def plot_student_results(student, scores, cohort_size):
    #  create the figure
    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')

    pos = np.arange(len(testNames))

    rects = ax1.barh(pos, [scores[k].percentile for k in testNames],
                     align='center',
                     height=0.5, color='m',
                     tick_label=testNames)

    ax1.set_title(student.name)

    ax1.set_xlim([0, 100])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

    # Plot a solid vertical gridline to highlight the median position
    ax1.axvline(50, color='grey', alpha=0.25)
    # set X-axis tick marks at the deciles
    cohort_label = ax1.text(.5, -.07, 'Cohort Size: {0}'.format(cohort_size),
                            horizontalalignment='center', size='small',
                            transform=ax1.transAxes)

    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()

    scoreLabels = [format_score(scores[k].score, k) for k in testNames]

    # set the tick locations
    ax2.set_yticks(pos)
    # make sure that the limits are set equally on both yaxis so the
    # ticks line up
    ax2.set_ylim(ax1.get_ylim())

    # set the tick labels
    ax2.set_yticklabels(scoreLabels)

    ax2.set_ylabel('Test Scores')

    ax2.set_xlabel(('Percentile Ranking Across '
                    '{grade} Grade {gender}s').format(
        grade=attach_ordinal(student.grade),
        gender=student.gender.title()))

    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = int(rect.get_width())

        rankStr = attach_ordinal(width)
        # The bars aren't wide enough to print the ranking inside
        if (width < 5):
            # Shift the text to the right side of the right edge
            xloc = width + 1
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = 0.98 * width
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2.0
        label = ax1.text(xloc, yloc, rankStr, horizontalalignment=align,
                         verticalalignment='center', color=clr, weight='bold',
                         clip_on=True)
        rect_labels.append(label)

    # make the interactive mouse over give the bar title
    ax2.fmt_ydata = format_ycursor
    # return all of the artists created
    return {'fig': fig,
            'ax': ax1,
            'ax_right': ax2,
            'bars': rects,
            'perc_labels': rect_labels,
            'cohort_label': cohort_label}


def plotTest010002():
    student = Student('Johnny Doe', 2, 'boy')
    scores = dict(zip(testNames,
                      (Score(v, p) for v, p in
                       zip(['7', '48', '12:52', '17', '14'],
                           np.round(np.random.uniform(0, 1,
                                                      len(testNames)) * 100, 0)))))
    cohort_size = 62  # The number of other 2nd grade boys

    arts = plot_student_results(student, scores, cohort_size)
    plt.show()


def plotTest011():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def plotTest012():
    data = [[66386, 174296, 75131, 577908, 32015],
            [58230, 381139, 78045, 99308, 160454],
            [89135, 80552, 152558, 497981, 603535],
            [78415, 81858, 150656, 193263, 69638],
            [139361, 331509, 343164, 781380, 52269]]

    columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    values = np.arange(0, 2500, 500)
    value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')

    plt.show()


def plotTest013():
    # Load a numpy record array from yahoo csv data with fields date, open, close,
    # volume, adj_close from the mpl-data/example directory. The record array
    # stores the date as an np.datetime64 with a day unit ('D') in the date column.
    with cbook.get_sample_data('goog.npz') as datafile:
        price_data = np.load(datafile)['price_data'].view(np.recarray)
    price_data = price_data[-250:]  # get the most recent 250 trading days

    delta1 = np.diff(price_data.adj_close) / price_data.adj_close[:-1]

    # Marker size in units of points^2
    volume = (15 * price_data.volume[:-2] / price_data.volume[0]) ** 2
    close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

    fig, ax = plt.subplots()
    ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

    ax.set_xlabel(r'$\Delta_i$', fontsize=15)
    ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
    ax.set_title('Volume and percent change')

    ax.grid(True)
    fig.tight_layout()

    plt.show()


def plotTest014():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = plt.plot(t, s, lw=2, color='red')
    plt.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp * np.sin(2 * np.pi * freq * t))
        fig.canvas.draw_idle()

    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        sfreq.reset()
        samp.reset()

    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)

    plt.show()


def plotTest015():
    x = np.linspace(0, 1, 500)
    y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

    fig, ax = plt.subplots()

    ax.fill(x, y, zorder=10)
    ax.grid(True, zorder=5)

    x = np.linspace(0, 2 * np.pi, 500)
    y1 = np.sin(x)
    y2 = np.sin(3 * x)

    fig, ax = plt.subplots()
    ax.fill(x, y1, 'b', x, y2, 'r', alpha=0.3)
    plt.show()


def plotTest016():
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    # Load a numpy record array from yahoo csv data with fields date, open, close,
    # volume, adj_close from the mpl-data/example directory. The record array
    # stores the date as an np.datetime64 with a day unit ('D') in the date column.
    with cbook.get_sample_data('goog.npz') as datafile:
        r = np.load(datafile)['price_data'].view(np.recarray)
    # Matplotlib works better with datetime.datetime than np.datetime64, but the
    # latter is more portable.
    date = r.date.astype('O')

    fig, ax = plt.subplots()
    ax.plot(date, r.adj_close)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    datemin = datetime.date(date.min().year, 1, 1)
    datemax = datetime.date(date.max().year + 1, 1, 1)
    ax.set_xlim(datemin, datemax)

    # format the coords message box
    def price(x):
        return '$%1.2f' % x

    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = price
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    plt.show()


def plotTest017():
    # Data for plotting
    t = np.arange(0.01, 20.0, 0.01)

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # log y axis
    ax1.semilogy(t, np.exp(-t / 5.0))
    ax1.set(title='semilogy')
    ax1.grid()

    # log x axis
    ax2.semilogx(t, np.sin(2 * np.pi * t))
    ax2.set(title='semilogx')
    ax2.grid()

    # log x and y axis
    ax3.loglog(t, 20 * np.exp(-t / 10.0), basex=2)
    ax3.set(title='loglog base 2 on x')
    ax3.grid()

    # With errorbars: clip non-positive values
    # Use new data for plotting
    x = 10.0 ** np.linspace(0.0, 2.0, 20)
    y = x ** 2.0

    ax4.set_xscale("log", nonposx='clip')
    ax4.set_yscale("log", nonposy='clip')
    ax4.set(title='Errorbars go negative')
    ax4.errorbar(x, y, xerr=0.1 * x, yerr=5.0 + 0.75 * y)
    # ylim must be set after errorbar to allow errorbar to autoscale limits
    ax4.set_ylim(ymin=0.1)

    fig.tight_layout()
    plt.show()


def plotTest018():
    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r

    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r)
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()


def plotTest019():
    # Make some fake data.
    a = b = np.arange(0, 3, .02)
    c = np.exp(a)
    d = c[::-1]

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(a, c, 'k--', label='Model length')
    ax.plot(a, d, 'k:', label='Data length')
    ax.plot(a, c + d, 'k', label='Total message length')

    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')

    plt.show()


# Selection of features following "Writing mathematical expressions" tutorial
mathtext_titles = {
    0: "Header demo",
    1: "Subscripts and superscripts",
    2: "Fractions, binomials and stacked numbers",
    3: "Radicals",
    4: "Fonts",
    5: "Accents",
    6: "Greek, Hebrew",
    7: "Delimiters, functions and Symbols"}
n_lines = len(mathtext_titles)

# Randomly picked examples
mathext_demos = {
    0: r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = "
       r"U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} "
       r"\int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ "
       r"U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_"
       r"{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$",

    1: r"$\alpha_i > \beta_i,\ "
       r"\alpha_{i+1}^j = {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau},\ "
       r"\ldots$",

    2: r"$\frac{3}{4},\ \binom{3}{4},\ \stackrel{3}{4},\ "
       r"\left(\frac{5 - \frac{1}{x}}{4}\right),\ \ldots$",

    3: r"$\sqrt{2},\ \sqrt[3]{x},\ \ldots$",

    4: r"$\mathrm{Roman}\ , \ \mathit{Italic}\ , \ \mathtt{Typewriter} \ "
       r"\mathrm{or}\ \mathcal{CALLIGRAPHY}$",

    5: r"$\acute a,\ \bar a,\ \breve a,\ \dot a,\ \ddot a, \ \grave a, \ "
       r"\hat a,\ \tilde a,\ \vec a,\ \widehat{xyz},\ \widetilde{xyz},\ "
       r"\ldots$",

    6: r"$\alpha,\ \beta,\ \chi,\ \delta,\ \lambda,\ \mu,\ "
       r"\Delta,\ \Gamma,\ \Omega,\ \Phi,\ \Pi,\ \Upsilon,\ \nabla,\ "
       r"\aleph,\ \beth,\ \daleth,\ \gimel,\ \ldots$",

    7: r"$\coprod,\ \int,\ \oint,\ \prod,\ \sum,\ "
       r"\log,\ \sin,\ \approx,\ \oplus,\ \star,\ \varpropto,\ "
       r"\infty,\ \partial,\ \Re,\ \leftrightsquigarrow, \ \ldots$"}


def doall():
    # Colors used in mpl online documentation.
    mpl_blue_rvb = (191. / 255., 209. / 256., 212. / 255.)
    mpl_orange_rvb = (202. / 255., 121. / 256., 0. / 255.)
    mpl_grey_rvb = (51. / 255., 51. / 255., 51. / 255.)

    # Creating figure and axis.
    plt.figure(figsize=(6, 7))
    plt.axes([0.01, 0.01, 0.98, 0.90], facecolor="white", frameon=True)
    plt.gca().set_xlim(0., 1.)
    plt.gca().set_ylim(0., 1.)
    plt.gca().set_title("Matplotlib's math rendering engine",
                        color=mpl_grey_rvb, fontsize=14, weight='bold')
    plt.gca().set_xticklabels("", visible=False)
    plt.gca().set_yticklabels("", visible=False)

    # Gap between lines in axes coords
    line_axesfrac = (1. / (n_lines))

    # Plotting header demonstration formula
    full_demo = mathext_demos[0]
    plt.annotate(full_demo,
                 xy=(0.5, 1. - 0.59 * line_axesfrac),
                 xycoords='data', color=mpl_orange_rvb, ha='center',
                 fontsize=20)

    # Plotting features demonstration formulae
    for i_line in range(1, n_lines):
        baseline = 1 - (i_line) * line_axesfrac
        baseline_next = baseline - line_axesfrac
        title = mathtext_titles[i_line] + ":"
        fill_color = ['white', mpl_blue_rvb][i_line % 2]
        plt.fill_between([0., 1.], [baseline, baseline],
                         [baseline_next, baseline_next],
                         color=fill_color, alpha=0.5)
        plt.annotate(title,
                     xy=(0.07, baseline - 0.3 * line_axesfrac),
                     xycoords='data', color=mpl_grey_rvb, weight='bold')
        demo = mathext_demos[i_line]
        plt.annotate(demo,
                     xy=(0.05, baseline - 0.75 * line_axesfrac),
                     xycoords='data', color=mpl_grey_rvb,
                     fontsize=16)

    for i in range(n_lines):
        s = mathext_demos[i]
        print(i, s)
    plt.show()


def plotTest020():
    if '--latex' in sys.argv:
        # Run: python mathtext_examples.py --latex
        # Need amsmath and amssymb packages.
        fd = open("mathtext_examples.ltx", "w")
        fd.write("\\documentclass{article}\n")
        fd.write("\\usepackage{amsmath, amssymb}\n")
        fd.write("\\begin{document}\n")
        fd.write("\\begin{enumerate}\n")

        for i in range(n_lines):
            s = mathext_demos[i]
            s = re.sub(r"(?<!\\)\$", "$$", s)
            fd.write("\\item %s\n" % s)

        fd.write("\\end{enumerate}\n")
        fd.write("\\end{document}\n")
        fd.close()

        subprocess.call(["pdflatex", "mathtext_examples.ltx"])
    else:
        doall()


def plotTest021():
    with plt.xkcd():
        # Based on "Stove Ownership" from XKCD by Randall Monroe
        # http://xkcd.com/418/

        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.xticks([])
        plt.yticks([])
        ax.set_ylim([-30, 10])

        data = np.ones(100)
        data[70:] -= np.arange(30)

        plt.annotate(
            'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
            xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

        plt.plot(data)

        plt.xlabel('time')
        plt.ylabel('my overall health')
        fig.text(
            0.5, 0.05,
            '"Stove Ownership" from xkcd by Randall Monroe',
            ha='center')

        # Based on "The Data So Far" from XKCD by Randall Monroe
        # http://xkcd.com/373/

        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        ax.bar([0, 1], [0, 100], 0.25)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, 1])
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([0, 110])
        ax.set_xticklabels(['CONFIRMED BY\nEXPERIMENT', 'REFUTED BY\nEXPERIMENT'])
        plt.yticks([])

        plt.title("CLAIMS OF SUPERNATURAL POWERS")

        fig.text(
            0.5, 0.05,
            '"The Data So Far" from xkcd by Randall Monroe',
            ha='center')

    plt.show()


def plotTest022():
    np.random.seed(19680801)
    data = np.random.randn(2, 100)

    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data[0])
    axs[1, 0].scatter(data[0], data[1])
    axs[0, 1].plot(data[0], data[1])
    axs[1, 1].hist2d(data[0], data[1])

    plt.show()


def plotTest023():
    print(" Nothing to show ")


def plotTest024():
    return


def plotTest025():
    return


def plotTest026():
    return


def plotTest027():
    return


def plotTest028():
    return


def plotTest029():
    return


if __name__ == "__main__":
    plotTest001()
    plotTest002()
    plotTest003001()
    plotTest003002()
    plotTest003003()
    plotTest003004()
    plotTest004()
    plotTest005()
    plotTest006()
    plotTest007()
    plotTest008()
    plotTest009()
    plotTest010()
    plotTest010002()
    plotTest011()
    plotTest012()
    plotTest013()
    plotTest014()
    plotTest015()
    plotTest016()
    plotTest017()
    plotTest018()
    plotTest019()
    plotTest020()
    plotTest021()
    plotTest022()
    plotTest023()

    plotTest024()
    plotTest025()
    plotTest026()
    plotTest027()
    plotTest028()
    plotTest029()
