'''
'''
import astrologs.astrologs as Astrologs


def TinkerGroup(): 
    ''' create postprocessed catalogs for Jeremy's group catalog 
    '''
    for mlim in ['9.7', '10.1', '10.5']: 
        tinker = Astrologs.TinkerGroup(mlim=mlim) 
        tinker._construct(silent=False) 

        tink = Astrologs.Astrologs('tinkergroup', mlim=mlim)
        print(tink.meta) 
    return None 


if __name__=='__main__': 
    TinkerGroup() 
