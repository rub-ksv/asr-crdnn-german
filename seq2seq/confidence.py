'''
This code implements
BOOTSTRAP ESTIMATES FOR CONFIDENCE INTERVALS
IN ASR PERFORMANCE EVALUATION
M. Bisani and H. Ney
and compares the results of our German speech recognition systems
'''

import numpy as np
from glob2 import glob

def get_confidence_interval(X, B=10**3, alpha=0.05):
    segments, dim = X.shape
    assert segments>2
    assert dim==2, 'Please provide X in the form of (n_1,e_1),...,(n_s,e_s)'

    idb = np.random.choice(range(X.shape[0]), size=B*segments)
    X_star_b = np.take(X, idb, axis=0).reshape(B, segments, -1)
    err_sum = np.sum(X_star_b[:,:,1], axis=1)
    word_sum = np.sum(X_star_b[:,:,0], axis=1)
    W_star_b = err_sum / word_sum
    W_boot = np.sum(W_star_b) / B

    diff = W_star_b - W_boot
    diff = diff * diff

    se_boot = np.sqrt(diff.sum() / (B - 1))
    W_sorted = np.sort(W_star_b)
    ret = {'confidence interval': (W_sorted[int(alpha * B)], W_sorted[int((1 - alpha) * B)]),
           'W_star_b': W_star_b,
           'W_boot': W_boot}
    return ret

def get_probability_of_improvement(X, Y, B=10**4):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == Y.shape[1]
    assert any(X[:,1] == Y[:,1]) # same segment lengths

    Z = X.copy()
    Z[:,1] = X[:,1] - Y[:,1]
    
    delta_W_star_b = get_confidence_interval(Z, B)['W_star_b']
    poi_boot = np.sum(delta_W_star_b < 0) / B
    return poi_boot

def parse_wer_line(line):
    ids = line.find('%WER')
    if ids >= 0:
        ids = line.find('[') 
        ide = line.find(']')
        if ids >= 0 and ide >= 0:
            line = line[ids + 1:ide].strip()
            e, n = line.split(',')[0].split('/')
            return(int(e), int(n))
    return None

if __name__=='__main__':
    results = {}
    for path in glob('./results/**/42'):
        data = []
        total_e = 0
        total_n = 0
        for wer_file in glob(f'{path}/wer_*.txt'):
            for ids, line in enumerate(open(wer_file,'r')):
                ret = parse_wer_line(line)
                if ret is not None:
                    e, n = ret
                    if ids == 0:
                        total_e += e
                        total_n += n
                    else:
                        data.append((n, e))
        data = np.array(data)
        results[path] = {'e': total_e, 'n': total_n, 'data': data}

        assert data[:,0].sum() == total_n
        assert data[:,1].sum() == total_e
    for k1 in results.keys():
        for k2 in results.keys():
            if k1 == k2:
                continue
            try:
                print('Probability that', k1, 'is better than', k2, 'is', get_probability_of_improvement(results[k1]['data'],results[k2]['data']))
                print('WER:',k1,results[k1]['e'] / results[k2]['n'])
                print()
            except:
                pass
