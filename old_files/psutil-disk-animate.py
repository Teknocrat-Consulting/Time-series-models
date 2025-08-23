import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import psutil
import threading
import time 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

df = pd.DataFrame()

def animate(j, xs, ys):

	t1 = dt.datetime.now().strftime('%M:%S') 
	xs.append(t1)

	c1 = psutil.disk_usage('/').used
	ys.append('%.3f'%(int(c1)/1000000000))

	xs = xs[-30:]
	ys = ys[-30:]

	# Draw x and y lists
	ax.clear()
	plt.xticks(rotation=45)
	ax.plot(xs, ys, label="Disk Usage")

	plt.title('Disk Usage over Time')
	plt.ylabel("space in GB")
	plt.xlabel('Time in Seconds')
	ax.legend(loc='upper left',fontsize=12)


if __name__ == "__main__":

	ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)

	plt.show()
	df['time'] = xs
	df['usage'] = ys
	df.index = df['time']
	del df['time']
	df.to_csv('disk.csv')
