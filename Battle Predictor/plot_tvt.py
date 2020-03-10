import csv
import matplotlib.pyplot as plt

with open("tvt.csv") as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    predictor_marines = int(row[0])
    enemy_marines = int(row[1])
    score = int(row[2])
    
    plt.plot(predictor_marines, enemy_marines, "go" if score > 0 else "ro")
  
plt.ylabel("enemy marines")
plt.xlabel("predictor marines")
plt.ylim(0, 11)
plt.xlim(0, 11)

plt.show()
