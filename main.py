# Original code by Jarod Najera 862179022
# small dataset 109
# large dataset 64
import nearest_neighbor as nn
import numpy
import time

if __name__ == '__main__':
    # load data
    file = input('Input file name:\n')
    print('')
    data = numpy.loadtxt(file)
    data = numpy.array(data)

    search_choice = int(input('Input 0 for Forward Elimination or 1 for Backwards Elimination\n'))
    print('')
    
    # Perform forward_elimination()
    if not search_choice:
        print('Performing Forward Elimination...')
        # Calculate runtime
        start = time.time()
        nn.forward_elimination(data)
        end = time.time()
        runtime = end - start

        if runtime < 60:
            print(f'Forward Elimination took {runtime:.2f} seconds')
        else:
            print(f'Forward Elimination took {runtime/60:.2f} minutes')
    else:
        print('Performing Backward Elimination...')
        # Calculate runtime
        start = time.time()
        nn.backward_elimination(data)
        end = time.time()
        runtime = end - start

        if runtime < 60:
            print(f'Backward Elimination took {runtime:.2f} seconds')
        else:
            print(f'Backward Elimination took {runtime/60:.2f} minutes')