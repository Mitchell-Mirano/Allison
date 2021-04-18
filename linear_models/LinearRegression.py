import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x=np.linspace(1,10,100)
y= 2*x +2 + 3 + np.random.randn(100)


dic={'x':x,'y':y}
df=pd.DataFrame(dic)

plt.Figure(figsize=(20,8))
plt.plot(x,y)