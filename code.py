
p, r, th = precision_recall_curve(util.rel, pp[:,1])

In [29]: len(p)
Out[29]: 3372

In [30]: import matplotlib.pyplot as plt

In [31]: plt.clf()

In [32]: plt.plot(r, p)
Out[32]: [<matplotlib.lines.Line2D at 0xb0c1e4c>]

In [33]: plt.show()

