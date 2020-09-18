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
                'tinkergroup': [TinkerGroup, "Jeremy's Group Catalog"], 
                'nsa': [NSAtlas, 'NASA-Sloan Atlas'], 
                'simsed': [SimSED, 'SEDs of simulations']
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
    
    def select(self, selection): 
        ''' selection a subset of the catalog based on selection criteria. This
        automatically removes any data columns that do not match the selection
        criteria dimensions so be careful. 

        :param selection: 
            Ngal dimensional boolean array specifying the galaxies you want to
            select. 

        '''
        if np.sum(selection) == 0: 
            raise ValueError("You're not keeping any galaxies") 
        
        Ndim = len(selection) 

        for k in self.data.keys(): 
            if self.data[k].shape[0] == Ndim: 
                self.data[k] = self.data[k][selection] 
            else: 
                del self.data[k] 
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
            if isinstance(cat[k], np.chararray): 
                hdf5.create_dataset(k, data=cat[k].astype(np.string_))
            else: 
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
            return None 
        
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
        cat['redshift']     = cz/2.998e5
        des['redshift']     = ['', 'redshift'] 
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


class NSAtlas(Catalog): 
    ''' NASA-Sloan Atlas

    references
    ----------
    * http://nsatlas.org/data
    '''
    def __init__(self): 
        super().__init__()
        self.file = self._File() # name  

    def options(self): 
        msg = '\n'.join([
            "",
            "NASA-Sloan Atlas where we relabel a handful of commonly used columns", 
            ""])
        print(msg)
        return None 

    def _File(self): 
        ''' file name of postprocessed catalog 
        '''
        name = os.path.join(
                os.environ.get('ASTROLOGS_DIR'), 'nsa', 'nsa.hdf5') 
        return name 

    def _construct(self, overwrite=False, silent=True): 
        ''' construct postprocessed catalogs from the NSA catalog .fits file 
        '''
        if not overwrite and os.path.isfile(self.file): 
            print("%s already exists; to overwrite specify `overwrite=True`" % self.file) 
            return None 
        
        from astropy.io import fits as Fits

        ffits = os.path.join(os.path.dirname(self.file), 'nsa_v0_1_2.fits') 
        
        # read in .prob file, which has the following data columns
        nsa = Fits.open(ffits)[1].data 

        cat, des = {}, {} 
        for k in nsa.dtype.names:
            cat[k] = nsa[k]
            des[k] = ['', 'see NSA description'] 
        
        cat['ra']           = nsa['RA'] 
        des['ra']           = ['deg', 'right ascension'] 
        cat['dec']          = nsa['DEC'] 
        des['dec']          = ['deg', 'declination'] 
        cat['redshift']     = nsa['Z']
        des['redshift']     = ['', 'redshift'] 
        cat['M_r']          = nsa['ABSMAG'][:,4]
        des['M_r']          = ['mag', 'r-band absolute magnitude from k-corrections']
        cat['M_g']          = nsa['ABSMAG'][:,3] 
        des['M_g']          = ['mag', 'g-band absolute magnitude from k-corrections']
        cat['Dn4000']       = nsa['D4000'] 
        des['Dn4000']       = ['', 'Dn4000']
        cat['M_star']       = nsa['MASS'] * 0.7**2 
        des['M_star']       = ['Msun', 'kcorrect stellar mass'] 
        cat['log.M_star']   = np.log10(cat['M_star'])
        des['log.M_star']   = ['dex', 'log stellar mass'] 
    
        if not silent: print('caluclating UV and Halpha based SFRs') 
        sfr_uv, sfr_ha = self._SFRs_for_nsa(cat) 
        cat['sfr_uv'] = sfr_uv
        des['sfr_uv'] = ['Msun/yr', 'UV SFR'] 
        cat['log.sfr_uv'] = np.log10(sfr_uv)
        des['log.sfr_uv'] = ['dex', 'log( UV SFR )'] 

        cat['sfr_ha'] = sfr_ha 
        des['sfr_ha'] = ['Msun/yr', 'Halpha SFR'] 
        cat['log.sfr_ha'] = np.log10(sfr_ha) 
        des['log.sfr_ha'] = ['dex', 'log( Halpha SFR )'] 

        # save to hdf5 
        self._save_to_hdf5(cat, self.file, meta=des, silent=silent)     
        return None 

    def _SFRs_for_nsa(self, nsa): 
        ''' calculate differnet types of SFRs (UV-based, Halpha-based) for the NSA catalog. 

        :param nsa: 
            nsa catalog dictionary 
        '''
        # calculate UV SFRs
        fuv         = nsa['NMGY'][:,0] # nanomaggies
        fuv_kcorr   = nsa['KCORRECT'][:,0] # kcorrect
        fuv_jansky  = self._jansky(fuv, fuv_kcorr)

        sfr_uv = self._SFR_UV(nsa['Z'],
                           nsa['ABSMAG'][:,0],
                           nsa['ABSMAG'][:,1],
                           nsa['ABSMAG'][:,4],
                           fuv_jansky)

        sfr_ha = self._SFR_Halpha(nsa['HAFLUX'], nsa['ZDIST'])

        return sfr_uv, sfr_ha

    def _jansky(self, flux, kcorrect):
        ''' get fluxes in Janskies from Nanomaggies:

        :param flux: 
            flux in nanomaggies

        :param kcorrect: 
            corresponding kcorrect 
        '''
        flux_in_Jy = flux*3631*(10.0**(-9.0))*(10**(kcorrect/(-2.5)))
        return flux_in_Jy

    def _SFR_UV(self, z, fmag, nmag, rmag, f_flux):
        ''' calculate UV star formation rates based on Salim+(2007) Eq. 5, 7, 8. 
        
        :param z:
            redshift

        :param fmag: 
            FUV absolute magnitude

        :param nmag: 
            NUV absolute magnitude

        :param rmag: 
            r-band absolute magnitude

        :param f_flux: 
            FUV flux in Janskies. See `_jansky` method for conversion 
        '''
        from astropy.cosmology import WMAP7
        fn = fmag - nmag
        opt = nmag - rmag   # N-r
        
        #Luminosity Distance
        dist = WMAP7.comoving_distance(z)
        ldist = (1+z) * dist.value
        
        #calculating Attenuation 'atten'
        atten = np.repeat(-999., len(fmag)) 

        case1 = np.where((opt > 4.) & (fn < 0.95))
        atten[case1] = 3.32*fn[case1] + 0.22
        case2 = np.where((opt > 4.) & (fn >= 0.95))
        atten[case2] = 3.37
        case3 = np.where((opt <= 4.) & (fn < 0.9))
        atten[case3] = 2.99*fn[case3] + 0.27
        case4 = np.where((opt <= 4.) & (fn >= 0.9))
        atten[case4] = 2.96

        #if opt >= 4.0:
        #    if fn < 0.95:
        #        atten = 3.32*fn + 0.22
        #    else:
        #        atten = 3.37
        #else:
        #    if fn < 0.90:
        #        atten = 2.99*fn +0.27
        #    else:
        #        atten = 2.96

        lum = 4.*np.pi*(ldist**2.0)*(3.087**2.0)*(10**(25.0 +(atten/2.5)))*f_flux  #Luminosity
        sfr = 1.08*(10**(-28.0))*np.abs(lum)
        return sfr

    def _SFR_Halpha(self, haflux, zdist): 
        ''' calculate SFR based on Halpha flux 

        :param haflux: 
            Halpha flux in units of 1e-17 erg/s/cm^2

        :param zdist: 
            Distance estimate using pecular velocity model of Willick et al. (1997); multiply by c/H0 for Mpc
        '''
        from astropy import units as U 
        from astropy import constants as Const

        ha_flux = haflux * 1e-17 * U.erg/U.s/U.cm**2
        H0 = 70. * U.km/U.s/U.Mpc

        ha_flux *= 4. * np.pi * (zdist * Const.c / H0)**2

        sfr = ha_flux.to(U.erg/U.s) /(10.**41.28)
        return sfr.value 


class SimSED(Catalog):
    ''' Catalogs with SEDs generated from SFH and ZH of simulated galaxies in
    specified simulation 

    '''
    def __init__(self, sim=None): 
        super().__init__()

        if sim is None: 
            self.options() 
            raise ValueError('please specify kwarg `sim`') 

        self.sim = sim 
        self.file = self._File(self.sim) # name  

    def options(self): 
        msg = '\n'.join([
            "",
            "Galaxy formation simulation with SEDs constructed from SFH", 
            "and ZH", 
            "", 
            "Specify the simulation using kwarg `sim`. The follow", 
            "simulations are available:", 
            "  `sim = 'simba'`", 
            "  `sim = 'tng'`", 
            "  `sim = 'eagle'`", 
            ""])
        print(msg)
        return None 

    def _File(self, sim): 
        ''' file name of postprocessed catalog 
        '''
        if sim not in ['simba', 'tng', 'eagle']: 
            self.options()
            raise ValueError('kwarg `sim` provided is not one of the options') 

        name = os.path.join(
                os.environ.get('ASTROLOGS_DIR'), 
                'simsed', 
                'simsed.%s.hdf5' % sim)
        return name 

    def _construct(self, overwrite=False, silent=True): 
        ''' these catalogs were constructed as part of the `galpopfm` project: 
        https://github.com/IQcollaboratory/galpopFM/blob/68133cf04d97276284b32eec2eb26c05a3db1f38/run/_sed.py

        todo
        ----
        * include metadata for the data columns 
        '''
        if not overwrite and os.path.isfile(self.file): 
            print("%s already exists; to overwrite specify `overwrite=True`" % self.file) 
            return None 
        raise ValueError
        return None 
