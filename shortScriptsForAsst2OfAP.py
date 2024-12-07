import math
 
for i in range(36):
    angle=i*10
    rad=math.radians(angle)
    print(f'{i}: [{math.cos(rad)}+{(math.sqrt(2-2*math.cos(rad)))}*cos(t),{math.sin(rad)}+{(math.sqrt(2-2*math.cos(rad)))}*sin(t),0]')