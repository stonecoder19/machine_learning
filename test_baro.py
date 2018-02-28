BARO_TAB_SIZE = 21
EstAlt = 0
baroHistTab = [0 for _ in range(21)]
baroHistIdx = 0
baroHigh = 0
BaroAlt = 50

while(1):
 
  baroHistTab[baroHistIdx] = BaroAlt/10
  baroHigh += baroHistTab[baroHistIdx]
  baroHigh -= baroHistTab[(baroHistIdx + 1)%BARO_TAB_SIZE]
  
  baroHistIdx+=1
  if (baroHistIdx == BARO_TAB_SIZE):
                baroHistIdx = 0


  EstAlt = EstAlt*0.6 + (baroHigh*10.0/(BARO_TAB_SIZE - 1))*0.4  
  
  print(baroHistTab)
  print(baroHigh)
  print(EstAlt)  
