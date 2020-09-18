'''
'''
import astrologs.astrologs as Astrologs


def TinkerGroup(): 
    ''' create postprocessed catalogs for Jeremy's group catalog 
    '''
    for mlim in ['9.7', '10.1', '10.5']: 
        tinker = Astrologs.TinkerGroup(mlim=mlim) 
        tinker._construct(silent=False) 

        alog = Astrologs.Astrologs('tinkergroup', mlim=mlim)
        print(alog.meta) 
    return None 


def NSAtlas(): 
    nsa = Astrologs.NSAtlas()
    nsa._construct(silent=False) 
        
    alog = Astrologs.Astrologs('nsa')
    print(alog.meta) 
    return None 


if __name__=='__main__': 
    TinkerGroup() 
    NSAtlas() 
