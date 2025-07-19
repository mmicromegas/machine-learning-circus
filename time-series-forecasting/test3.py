from statsmodels.datasets import co2
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
data = co2.load().data
data = data.resample('ME').mean().ffill()

print(type(data))
print(data.head())

from statsmodels.tsa.seasonal import STL
res = STL(data).fit()
res.plot()
plt.show()

