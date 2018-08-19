from map import map

def main(S,ops=None,u=None,sv=None):
    isort2,V = map(S.T,ops,u,sv)
    Sm = S - S.mean(axis=1)
    Sm = gaussian_filter1d(Sm,5,axis=1)
    isort1,V = map(Sm,ops,u,sv)
    ns = 0.02 * Sm.shape[0]
    Sm = gaussian_filter1d(Sm[isort1,:],ns,axis=1)
    return Sm,isort1,isort2

if __name__ == "__main__":
    main()
