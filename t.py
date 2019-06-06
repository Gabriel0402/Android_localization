import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

size =[400,500,600,700,800,900,999]
df = pd.read_csv('t2.csv',index_col=0)
df['data size']=pd.Series(size)
df.set_index('data size',inplace=True)
df.plot.line(colormap='gist_rainbow')
# df.plot.line()
plt.show()