from test1 import PINN
from test1 import answer
current_Stress,pos,plot,M=PINN(200,80,200,"Five-Degree")
#将pos赋值给D0,D1,D2……
response=answer(pos,"Five-Degree",current_Stress,M)
print(response)