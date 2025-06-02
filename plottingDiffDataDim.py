import matplotlib.pyplot as plt
import csv

def plot_avg_time_vs_num_satellites(num_satellites, avg_times, data_size):
    plt.figure()
    plt.plot(num_satellites, avg_times, marker='o')
    plt.xlabel('Number of Satellites')
    plt.ylabel('Average Time (s)')
    plt.title(f'Average Propagation Time vs Number of Satellites (Data Size: {data_size} bits)')
    plt.grid(True)
    filename = f'avg_time_vs_num_satellites_data_size_{data_size}.png'
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to avoid displaying it

    # Write the results to a text file
    with open(f"TimevsNum_data_size_{data_size}.txt", "w") as output:
        output.write(str(num_satellites))
        output.write("\n")
        output.write(str(avg_times))

    # Write the results to a CSV file
    with open(f'MyList_data_size_{data_size}.csv', 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(num_satellites)
        wr.writerow(avg_times)
