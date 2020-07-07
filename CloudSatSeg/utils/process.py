from tqdm import tqdm
from multiprocessing import Pool

def argwrapper(args):
    return args[0](*args[1:])


def imap_unordered_bar(func, args, n_processes=15):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list