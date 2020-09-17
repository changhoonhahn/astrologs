'''



'''
import os 
import h5py
import numpy as np 


class Astrologs(object): 
    ''' class object for loading in various postprocessed catalogs 
    '''
    def __init__(self, catalog, **kwargs): 
        ''' load specified catalog 
        '''
        self._available_catalogs = {
                'tinkergroup': [TinkerGroup, "Jeremy's Group Catalog"]
                } 
        
        if catalog not in self._available_catalogs.keys(): 
            self.options() 
            raise ValueError("%s is not one of the available catalogs\n")

        cat_obj = self._available_catalogs[catalog][0](**kwargs)
        cat, meta = cat_obj.load() 

        # store data and meta data 
        self.data = cat
        self.meta = meta 

    def options(self): 
    
        avail = '\n*'.join(['%s : %s' % (k, self._available_catalogs[k][1]) for
            k in self._available_catalogs.keys()]) 

        msg = "\n currently available catalogs:\n %s \n" % avail 
        print(msg)
        return None 


class Catalog(object): 
    ''' parent class object for postprocessed catalog
    '''
    def __init__(self): 
        self.file = None # name of postprocessed hdf5 catalog 
        
        if os.environ.get('ASTROLOGS_DIR') is None: 
            raise ValueError("specify `ASTROLOGS_DIR` in bashrc or bash_profile") 

    def options(self): 
        ''' Some catalog objects will have multiple options (e.g. different M*
        cuts). This method will print a message explaining the options.
        '''
        msg = 'description of options here'
        print(msg) 
        return None 

    def load(self): 
        ''' load postprocessed hdf5 catalog into dictionary 
        '''
        if not os.path.isfile(self.file): 
            raise ValueError('%s does not exist') 

        return self._hdf5_to_dict(self.file) 

    def _construct(self): 
        ''' construct postprocessed hdf5 catalog 
        '''
        pass

    def _hdf5_to_dict(self, fhdf5): 
        ''' read hdf5 file and write to dictionary. 

        :param fhdf5: 
            hdf5 file name

        todo
        ----
        * implement fixes for reading strings and such 
        '''
        fcat = h5py.File(fhdf5, 'r')

        cat = {}
        for k in fcat.keys(): 
            cat[k] = fcat[k][...]

        # load attributes to meta data 
        meta = {}
        for k in fcat.attrs.keys(): 
            meta[k] = fcat.attrs[k] 
        return cat, meta 

    def _save_to_hdf5(self, cat, fhdf5, meta=None, silent=True): 
        ''' save python dictionary `cat` to hdf5
        '''
        if not silent: print('constructing %s' % fhdf5) 

        hdf5 = h5py.File(fhdf5, 'w') 

        for k in cat.keys(): 
            hdf5.create_dataset(k, data=cat[k]) 
        
        if meta is not None: 
            for k in meta.keys(): 
                hdf5.attrs[k] = meta[k] 

        hdf5.close() 
        return None 


class TinkerGroup(Catalog): 
    ''' Jeremy's group catalog 


    references
    ----------
    * https://cosmo.nyu.edu/~tinker/GROUP_FINDER/PCA_CATALOGS/
    * https://cosmo.nyu.edu/~tinker/GROUP_FINDER/PCA_CATALOGS/info.txt
    '''
    def __init__(self, mlim=None): 
        super().__init__()

        if mlim is None: 
            self.options() 
            raise ValueError('please specify kwarg `mlim`') 

        self.mlim = mlim  
        self.file = self._File(self.mlim) # name  

    def options(self): 
        msg = '\n'.join([
            "",
            "Jeremy's group catalogs are volume limited samples defined by", 
            "stellar mass and magnitude limits. There are 3 different M*", 
            "limits: 9.7, 10.1, 10.5. These roughly correspond to Mr cuts:", 
            "-17.5, -18.3, -19.0",
            "", 
            "To select one fo the three catalogs, specify kwarg `mlim`:", 
            "  `mlim = '9.7'`", 
            "  `mlim = '10.1'`", 
            "  `mlim = '10.5'`", 
            ""])
        print(msg)
        return None 

    def _File(self, mlim): 
        ''' file name of postprocessed catalog 
        '''
        if mlim not in ['9.7', '10.1', '10.5']: 
            self.options()
            raise ValueError('kwarg `mlim` provided is not one of the options') 

        name = os.path.join(
                os.environ.get('ASTROLOGS_DIR'), 
                'tinkergroup', 
                'tinker.groups.pca.M%s.hdf5' % mlim) 
        return name 

    def _construct(self, overwrite=False, silent=True): 
        ''' construct postprocessed catalogs from Jeremy's *.galdata_corr and *.prob
        files 

        *.prob is the galaxy catalog with probability of being a satellite
        (official stats assume >0.5 is a satellite).

        *.galdata_corr are galaxy properties. This file is in the same order
         as the *.prob file

        '''
        if not overwrite and os.path.isfile(self.file): 
            print("%s already exists; to overwrite specify `overwrite=True`" % self.file) 
        
        fprob = os.path.join(os.path.dirname(self.file), 
                'pca_groups_M%s.prob' % self.mlim)
        fgald = os.path.join(os.path.dirname(self.file),
                'pca_groups_M%s.galdata_corr' % self.mlim) 
        
        # read in .prob file, which has the following data columns
        # 1) foo
        # 2) gal id
        # 3) group id
        # 4) id of central galaxy
        # 5) r-band magnitude h=1
        # 6) P_sat (>0.5 means a satellite)
        # 7) halo mass [Msol/h]
        # 8) foo
        # 9) foo
        # 10) foo
        # 11) projected separation of gal from central, in units of Rhalo
        # 12) projected separation of gal in units of radians
        # 13) angular radius of halo
        galid, grpid, cenid, rmag, psat, mhalo, sep_rhalo, sep_rad, r_halo = \
                np.loadtxt(fprob, unpack=True, usecols=[1, 2, 3, 4, 5, 6, 10, 11, 12])

        # read in .galdata_corr file, which has the following data columns
        # this file is line matched to the prob file 
        # 1) foo
        # 2) gal id
        # 3) M_r
        # 4) M_g
        # 5) cz [km/s]
        # 6) Dn4000  (from MPA/JHU catalog)
        # 7) H_delta EW (same source)
        # 8) log sSFR (1/yr same source, but uses kcorrect stellar mass for "s")
        # 9) stellar mass (Msol/h^2, from kcorrect)
        # 10) ra
        # 11) dec
        # 12) velocity dispersion (from VAGC)
        # 13) signal to noise of spectrum (from VAGC)
        # 14) sersic index (from VAGC) 
        Mg, cz, Dn4000, Hdel_ew, logSSFR, Ms, ra, dec, vdisp, sn, sersic = \
                np.loadtxt(fgald, unpack=True, usecols=range(3,14)) 

        cat, des = {}, {} 
        cat['galid']        = galid 
        des['galid']        = ['', 'galaxy id']
        cat['groupid']      = grpid 
        des['groupid']      = ['', 'group id']
        cat['centralid']    = cenid
        des['centralid']    = ['', 'id of central galaxy']
        cat['M_r']          = rmag
        des['M_r']          = ['mag', 'r-band magnitude h=1']
        cat['M_g']          = Mg
        des['M_g']          = ['mag', 'g-band magnitude h=1']
        cat['P_sat']        = psat
        des['P_sat']        = ['', 'P_sat > 0.5 are satellite']
        cat['iscen']        = (psat <= 0.5) # is central 
        des['iscen']        = ['', 'True: central; False: satellite']
        cat['M_halo']       = mhalo
        des['M_halo']        = ['Msun/h', 'halo mass']
        cat['sep_rhalo']    = sep_rhalo
        des['sep_rhalo']    = ['R_halo', 'proj. sep. from central']
        cat['sep_rad']      = sep_rhalo
        des['sep_rad']      = ['radian', 'proj. sep. from central']
        cat['r_halo']       = r_halo
        des['r_halo']       = ['?', 'angular radius of halo'] 
        cat['cz']           = cz
        des['cz']           = ['km/s', 'c * redshift'] 
        cat['Dn4000']       = Dn4000
        des['Dn4000']       = ['', 'Dn4000 from MPAJHU']
        cat['H_delta_EW']   = Hdel_ew
        des['H_delta_EW']   = ['', 'H_delta EW MPAJHU'] 
        cat['log.ssfr']     = logSSFR
        des['log.ssfr']     = ['dex', 'MPAJHU but using kcorrect M*']
        cat['M_star']       = Ms
        des['M_star']       = ['Msun/h^2', 'kcorrect stellar mass'] 
        cat['log.M_star']   = np.log10(Ms)
        des['log.M_star']   = ['dex', 'log stellar mass'] 
        cat['ra']           = ra
        des['ra']           = ['deg', 'right ascension'] 
        cat['dec']          = dec
        des['dec']          = ['deg', 'declination'] 
        cat['vdisp']        = vdisp 
        des['vdisp']        = ['km/s', 'velocity dispersion from VAGC']
        cat['s2n']          = sn
        des['s2n']          = ['', 'signal-to-noise of spectrum from VAGC'] 
        cat['sersic']       = sersic
        des['sersic']       = ['', 'sersic index from VAGC'] 
        
        # save to hdf5 
        self._save_to_hdf5(cat, self.file, meta=des, silent=silent)     
        return None 
