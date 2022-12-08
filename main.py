# Original code by Jarod Najera 862179022
# small dataset 109
# large dataset 64
import nearest_neighbor as nn
import numpy

if __name__ == '__main__':
    # load data
    data = numpy.loadtxt('CS170_Small_Data__109.txt')
    data = numpy.array(data)

    nn.feature_search(data)