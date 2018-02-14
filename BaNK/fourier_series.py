import numpy as np


def cn(n):
   c = y*np.exp(-1j*2*n*np.pi*time/period)
   return c.sum()/c.size

def f(x, Nh):
   f = np.array([2*cn(i)*np.exp(1j*2*i*np.pi*x/period) for i in range(1,Nh+1)])
   return f.sum()

y2 = np.array([f(t,50).real for t in time])

plot(time, y)
plot(time, y2)