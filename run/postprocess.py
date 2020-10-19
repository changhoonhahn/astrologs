'''
'''
import astrologs.astrologs as Astrologs


def TinkerGroup(): 
    ''' create postprocessed catalogs for Jeremy's group catalog 
    '''
    for mlim in ['9.7', '10.1', '10.5']: 
        for cross_nsa in [False, True]: 
            tinker = Astrologs.TinkerGroup(mlim=mlim, cross_nsa=cross_nsa) 
            tinker._construct(silent=False) 

            alog = Astrologs.Astrologs('tinkergroup', mlim=mlim)
            print(alog.meta) 
    return None 


def VAGC():
    ''' postprocess NYU VAGC DR72BRIGHT34
    '''
    for cross_nsa in [False, True]: 
        vagc = Astrologs.VAGC(cross_nsa=cross_nsa)
        vagc._construct(silent=False) 

        alog = Astrologs.Astrologs('vagc') 
        print(alog.meta) 
    return None


def NSAtlas(): 
    for vagc_footprint in [False, True]: 
        nsa = Astrologs.NSAtlas(vagc_footprint=vagc_footprint)
        nsa._construct(silent=False) 
        
        alog = Astrologs.Astrologs('nsa')
        print(alog.meta) 
    return None 


if __name__=='__main__': 
    TinkerGroup() 
    #NSAtlas() 
    #VAGC()
