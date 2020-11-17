import math
import numpy as np
if __name__ == "__main__":

   a= np.zeros()
   a = math.log2(0.02) *0.02
   a += math.log2(0.1) *0.1
   a += math.log2(0.07) *0.07
   a += math.log2(0.002) *0.002
   a += math.log2(0.8) *0.8
   a += math.log2(0.008) * 0.008

   print(-a)
   print(0.8*1+0.1*2+0.07*3+0.08*4+0.008*5+0.002*5)

   0.02
   0.1
   0.07
   0.002
   0.8
   0.008
