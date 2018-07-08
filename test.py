# import Map
#
# mp=Map.map()
#
# mp.Matrix[0,1]=2
# mp.Matrix[0,2]=2
# mp.Matrix[0,3]=4
# mp.Matrix[0,0]=4
# mp.Matrix[1,1]=2
# mp.Matrix[1,2]=2
# mp.Matrix[1,3]=4
# mp.Matrix[1,0]=4
# mp.Matrix[2,1]=2
# mp.Matrix[2,2]=2
# mp.Matrix[2,3]=4
# mp.Matrix[2,0]=4
# mp.Matrix[3,1]=2
# mp.Matrix[3,2]=2
# mp.Matrix[3,3]=4
# mp.Matrix[3,0]=4
#
# ac=[0,1,2,3]
#
# while True:
#     print(mp.Matrix)
#     k=mp.move(ac[0])
#     print(k)
#     if k:
#         break
import numpy as np

a=np.zeros((2,3))
a[1,1]=1
print(np.max(a))
