import numpy as np


def match_timestamps(times_a, times_b):
    '''
    Match two lists of timestamps by closeness in time.
    inputs:
        times_a: list of ascending timestamps
        times_b: list of ascending timestamps
    outputs:
        matches_a: list of indices into times_b for each element of times_a
        matches_b: list of indices into times_a for each element of times_b
        diffs_a: list of differences between each element of times_a and its match in times_b
        diffs_b: list of differences between each element of times_b and its match in times_a
    '''
    i, j = 0, 0
    matches_a = [-1] * len(times_a)
    matches_b = [-1] * len(times_b)
    diffs_a = [float('inf')] * len(times_a)
    diffs_b = [float('inf')] * len(times_b)
    while i < len(times_a) and j < len(times_b):
        curdiff = abs(times_a[i] - times_b[j])
        if curdiff < diffs_a[i]:
            diffs_a[i] = curdiff
            matches_a[i] = j
        if curdiff < diffs_b[j]:
            diffs_b[j] = curdiff
            matches_b[j] = i
        if times_a[i] < times_b[j]:
            i += 1
        else:
            j += 1
    # fill in the rest
    if i < len(times_a):
        matches_a[i:] = [j - 1] * (len(times_b) - i)
        diffs_a[i:] = [abs(times_a[i] - times_b[-1])] * (len(times_a) - i)
    if j < len(times_b):
        matches_b[j:] = [i - 1] * (len(times_a) - j)
        diffs_b[j:] = [abs(times_a[-1] - times_b[j])] * (len(times_b) - j)
    return matches_a, matches_b, diffs_a, diffs_b


class ConcatMemmap(object):
    def __init__(self, dtype: np.dtype = np.float32):
        '''
        create an empty ConcatMemmap object

        Args:
            dtype(np.dtype): the datatype of the memmap

        Returns:
            ConcatMemmap: the empty ConcatMemmap object
        '''
        self.dtype = dtype
        self.data = []
        self.indmap = []
        self.accumulatedLength = [0]

    def append(self, data: np.memmap):
        '''
        append a memmap to the ConcatMemmap

        Args:
            data(np.ndarray): the data to append
        '''
        assert data.dtype == self.dtype, f"Expected data to have dtype {self.dtype}, got {data.dtype}"
        memmap_ind = len(self.data)
        self.data.append(data)
        self.indmap.extend([memmap_ind] * len(data))
        self.accumulatedLength.append(len(data) + self.accumulatedLength[memmap_ind])

    def __getitem__(self, index):
        '''
        get item from the ConcatMemmap.
        This is a nuanced operation because a slice can technically span multiple memmaps.
        Currently, we don't allow that here to preserve the time complexity signature of slicing

        Args:
            index: the index to get, can be int or slice

        Returns:
            np.ndarray: the data at the given index
        '''
        if isinstance(index, slice):
            if index.stop > len(self):
                raise IndexError("Index out of bounds")
            super_index = self.indmap[index.start]
            sub_index = slice(index.start - self.accumulatedLength[super_index],
                              index.stop - self.accumulatedLength[super_index],
                              index.step)
            if sub_index.stop > len(self.data[super_index]):
                raise IndexError(f"Slice {index} out of bounds, slicing across memmaps currently not supported")
        else:
            super_index = self.indmap[index]
            sub_index = index - self.accumulatedLength[super_index]
        return self.data[super_index][sub_index]

    def __len__(self):
        '''
        get the length of the ConcatMemmap

        Returns:
            int: the length of the ConcatMemmap
        '''
        return len(self.indmap)
