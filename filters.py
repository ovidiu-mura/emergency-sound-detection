

def filter_amps_below(signal, amp):
    # signal: signal to be filtered
    # amp: the threshold amplitude; the amps below this amp will be considered
    #
    a = []
    for i in range(1,238):
        if(i<237):
            b=0
            if(len(a)!=0):
                b = a[-1]
            if(signal[i]<b+amp):
                a.insert(i, signal[i])
    a[0] = 0
    return a
