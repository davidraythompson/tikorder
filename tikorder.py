# David R. Thompson
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any commercial
# use must be negotiated with the Office of Technology Transfer at the
# California Institute of Technology.  This software is controlled under the
# U.S. Export Regulations and may not be released to foreign persons without
# export authorization, e.g., a license, license exception or exemption.

from spectral.io import envi
import sys, argparse, json
from os import makedirs
from os.path import join as pathjoin, exists as pathexists
from scipy.linalg import LinAlgError, eig, svd, inv, block_diag
from numpy.linalg import slogdet
from scipy.optimize import least_squares
from scipy import signal
from scipy.interpolate import interp1d
import scipy as sp
from collections import OrderedDict
from numpy import matmul
import pylab as plt
import logging
import pylab as plt
import multiprocessing as mp


# Suppress warnings that don't come from us
import warnings
warnings.filterwarnings("ignore")



def srf(x, mu, sigma):
    """Spectral Response Function """
    u = (x-mu)/abs(sigma)
    y = (1.0/(sp.sqrt(2.0*sp.pi)*abs(sigma)))*sp.exp(-u*u/2.0)
    return y/y.sum()


def spectrumResample(x, wl, wl2, fwhm2=10, fill=False):
    """Resample a spectrum to a new wavelength / FWHM.
       I assume Gaussian SRFs"""
    return sp.array([x[sp.newaxis,:] @ srf(wl, wi, fwhmi/2.355)[:,sp.newaxis]
                     for wi, fwhmi in zip(wl2, fwhm2)]).reshape((len(wl2)))


class ConstraintViolation(Exception):
    def __init__(self, message):
        self.msg = message


class LibrarySpectrum():

    def __init__(self, config):
        """Parse a configuration object, get spectrum and constraints"""
        self.wl, self.rfl = sp.loadtxt(config['reflectance_file']).T
        if all(self.wl<100):
            self.wl = self.wl * 1000.0 # convert to nm
        self.rfl = self.rfl.reshape((len(self.rfl),))
        self.group     = config['group']
        self.name      = config['name']
        self.prior     = sp.array(config['prior'])
        self.mixing    = sp.array(config['mixing_fraction'])
        self.continuua = []
        self.cr        = [] # container for the actual library cr
        for feature in config['features']:
            self.continuua.append(feature['continuum'])
        self.n_intervals = len(self.continuua)
        for constraint in ['constraint_ctm_slope', 'constraint_ctm_left',
                           'constraint_ctm_right','constraint_ctm']:
            setattr(self, constraint, [])
            for feature in config['features']:
                if constraint in feature:
                    getattr(self,constraint).append(feature[constraint])
                else:
                    getattr(self,constraint).append([-9999.0, 9999.0])

    def resample(self, wl, fwhm):
        """Resample the spectrum to a new wavelength grid"""
        rfl = self.rfl.copy().reshape(self.wl.shape)
        self.rfl = spectrumResample(rfl, self.wl, wl, fwhm)
        self.wl = wl.copy()

    def ctmrm(self, rfl, plot=False):
        """Remove the continuua from a reflectance spectrum, raising
            exceptions if any constraint is violated.

            Input:
              rfl        - Reflectance spectrum.  Should have the same
                           wavelength sampling and size as this library
                           spectrum.
            Returns:
              refl_ctmrm - Continuum-removed reflectance across N features.
                           The format is a C x N array, where C is the total
                           number of wavelengths across all feature intervals
                           (concatenated one after the other) and N is the
                           number of feature intervals.  Each interval's
                           continuum-removed reflectance appears in the
                           relevant column, and the reflectances of all other
                           intervals in that column (representing other
                           features) are set to zero.
              ctm        - The continuum estimate in the original reflectance
                           units.  The format is a C x N array with te same
                           block-nonzero structure described above.
              ivals      - A binary matrix of size C x N (see above) with
                           ones in the "active" channels for each feature's
                           column, zeros elsewhere.
              idx        - An integer matrix representing the indices of
                           all feature intervals, concatentated together, in
                           the original reflectance spectrum.  In other words,
                           the mapping from channels in rfl_ctmrm to channels
                           in the original input, rfl.
            """
        rfl_ctmrm = [[] for q in range(self.n_intervals)]
        ctms      = [[] for q in range(self.n_intervals)]
        ivals     = [[] for q in range(self.n_intervals)]
        idx_ctmrm = []

        for i in range(self.n_intervals):

            # Locate continuum
            rctma, rctmb, lctma, lctmb = self.continuua[i]
            in_rct = sp.logical_and(self.wl>rctma, self.wl<rctmb)
            in_lct = sp.logical_and(self.wl>lctma, self.wl<lctmb)
            rctm_idx = int(sp.where(in_rct)[0].mean())
            lctm_idx = int(sp.where(in_lct)[0].mean())
            idx_ctmrm.extend(range(rctm_idx,lctm_idx+1))
            rct = rfl[in_rct].mean()
            lct = rfl[in_lct].mean()
            ct = (rct+lct)/2.0
            slope = rct/lct

            # Check constraints
            if not (rct > self.constraint_ctm_right[i][0] and \
                    rct < self.constraint_ctm_right[i][1]):
                raise ConstraintViolation('rct = %f'%rct)
            if not (lct > self.constraint_ctm_left[i][0] and \
                    lct < self.constraint_ctm_left[i][1]):
                raise ConstraintViolation('lct = %f'%rct)
            if not (slope > self.constraint_ctm_slope[i][0] and \
                    slope < self.constraint_ctm_slope[i][1]):
                raise ConstraintViolation('rct/lct = %f'%slope)
            if not (ct > self.constraint_ctm[i][0] and \
                    ct < self.constraint_ctm[i][1]):
                raise ConstraintViolation('ct = %f'%ct)

            # divide by local continuum across this interval
            rfl_ival = rfl[rctm_idx:(lctm_idx+1)]
            n_channels = len(rfl_ival)
            p = interp1d([0, n_channels-1], [rfl_ival[0], rfl_ival[-1]])
            ctm = p(range(n_channels))

            if plot:
                plt.plot(self.wl, rfl)
                plt.plot(self.wl[rctm_idx:(lctm_idx+1)], p(range(n_channels)))
                plt.show(block=True)

            for j in range(self.n_intervals):
                if i==j:
                    rfl_ctmrm[j].extend(rfl_ival/ctm)
                    ctms[j].extend(ctm)
                    ivals[j].extend(sp.ones(n_channels))
                else:
                    rfl_ctmrm[j].extend(sp.zeros(n_channels))
                    ctms[j].extend(sp.zeros(n_channels))
                    ivals[j].extend(sp.zeros(n_channels))

        return (sp.array(rfl_ctmrm).T, sp.array(ctms).T, sp.array(ivals).T,
                idx_ctmrm)

    def fit (self, rfl, uncert, plot=False):
        # Set up matrices
        obs_noise = pow(uncert.copy(), 2)
        if uncert.ndim < 2 or (uncert.shape[0] != uncert.shape[1]):
            obs_noise = sp.diag(obs_noise)
        rfl_ref, ctm_ref, ivals, idx = self.cr
        rfl_test, ctm_test, ivals, idx = self.ctmrm(rfl)
        rfl_test = rfl_test.sum(axis=1)
        ctm_test = ctm_test.sum(axis=1)

        # State vector has multipliers, offsets in that order (one per feature)
        K = sp.concatenate((rfl_ref, ivals), axis=1)
        offs_prior = sp.eye(ivals.shape[1])
        S_a = block_diag(self.prior, offs_prior)
        x_a = sp.zeros(S_a.shape[0])

        # Input uncertainty treats continuum removal, a linear transformation
        # It currently ignores uncertainty in the continuum placement itself!
        S_e = sp.array([obs_noise[i, idx] for i in idx])
        Q   = sp.eye(len(ctm_test)) * (1.0/ctm_test)
        S_e = Q.T @ S_e @ Q
        iS_e, iS_a = inv(S_e), inv(S_a)

        # Tikonov solution provides true posteriors
        x = x_a + inv(K.T @ iS_e @ K + iS_a) @ K.T @ iS_e @ (rfl_test - K @ x_a)
        S_hat = inv(K.T @ iS_e @ K + iS_a)
        rfl_hat = K @ x
        residual = rfl_hat - rfl_test
        sign, logdet = slogdet(S_e)
        Z = len(residual) * sp.log(2*sp.pi) + logdet
        nll = 0.5 * (residual @ iS_e @ residual + Z)
        corr = pow(sp.corrcoef(rfl_hat, rfl_test)[0, 1], 2)

        # Depth estimate is taken from the most certain measurement
        # In the future consider a Kalman-like update
        coeffs = x[:self.n_intervals]
        uncerts = sp.sqrt(sp.diag(S_hat))[:self.n_intervals]
        best = sp.argmin(uncerts)
        depth = coeffs[best]
        post  = uncerts[best]

        return depth, post, nll, corr



class Library():

    def __init__(self, config):

        self.lib = OrderedDict()
        for fn in config:
            for source in config['sources']:
                src = LibrarySpectrum(source)
                if src.group in self.lib:
                    self.lib[src.group].append(src)
                else:
                    self.lib[src.group] = [src]

        c, self.wl, self.fwhm = sp.loadtxt(config['wavelength_file']).T
        if all(self.wl < 100):
           self.wl = self.wl * 1000 # convert to nm
           self.fwhm = self.fwhm * 1000
        self.resample(self.wl, self.fwhm)
        self.cr() # calculate cr for lib instances

    def resample(self, wl, fwhm):
        for grp in self.lib:
            for i in range(len(self.lib[grp])):
                self.lib[grp][i].resample(wl, fwhm)
        self.wl = wl
        self.fwhm = fwhm

    def cr(self):
        for group in self.lib:
            for i in range(len(self.lib[group])):
                rfl = self.lib[group][i].rfl
                self.lib[group][i].cr = self.lib[group][i].ctmrm(rfl)

    def fit(self, data, plot=False):
        rfl = data[:, 0]
        uncert = data[:, 1]
        group_depths, group_posts, group_nlls, group_corrs = [], [], [], []
        for group, spectra in self.lib.items():
            depths, posts, nlls, models, corrs = [],[],[],[],[] # negative log likelihood of fitted model
            for spectrum in spectra:
                try:
                    depth, post, nll, corr = spectrum.fit(rfl, uncert, plot)
                    corrs.append(corr)
                    nlls.append(nll)
                    depths.append(depth * spectrum.mixing)
                    posts.append(post * spectrum.mixing)
                    models.append(spectrum.name)
                except ConstraintViolation as v:
                    continue
            if len(nlls) == 0:
               group_depths.append(0)
               group_posts.append(-9999)
               group_nlls.append(999999)
               group_corrs.append(0)
               continue
            best_fit_idx = sp.argmin(nlls)
            group_depths.append(depths[best_fit_idx])
            group_posts.append(posts[best_fit_idx])
            group_nlls.append(nlls[best_fit_idx])
            group_corrs.append(corrs[best_fit_idx])
        return (sp.array(group_depths), sp.array(group_posts),
                sp.array(group_nlls), sp.array(group_corrs))

    def nchan(self):
        return len(self.lib)

    def groups(self):
        return self.lib.keys()


def main():

    # Parse command line
    description = 'Spectroscopic Surface & Atmosphere Fitting'
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--level', default='INFO')
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', level=args.level)

    # Load a parallel Pool
    pool = mp.Pool(mp.cpu_count())

    # Load the configuration file.
    config = json.load(open(args.config_file, 'r'))
    logging.info('Loading library')
    lib = Library(config['library'])

    # Get image and wavelengths
    logging.info('Opening input data')
    rfl_inhdr    = str(config['input_reflectance']+'.hdr')
    uncert_inhdr = str(config['input_uncertainty']+'.hdr')
    depth_outhdr = str(config['output_depths']+'.hdr')
    post_outhdr  = str(config['output_posterior']+'.hdr')
    nll_outhdr   = str(config['output_likelihood']+'.hdr')
    corr_outhdr  = str(config['output_corr']+'.hdr')

    R = envi.open(rfl_inhdr)
    meta = R.metadata.copy()
    U = envi.open(uncert_inhdr)

    # Now that the input images are available, resample wavelengths
    if 'wavelength' in meta:
        wl = sp.array([float (w) for w in meta['wavelength']])
        if all(wl<100): wl = wl * 1000.0
    else:
        wl = lib.wl.copy()
    if 'fwhm' in meta:
        fwhm = sp.array([float (f) for f in meta['fwhm']])
        if all(fwhm<0.1): fwhm = fwhm * 1000.0
    else:
        fwhm = sp.ones(wl.shape) * (wl[1]-wl[0])
    lib.resample(wl, fwhm)

    # Create output images
    meta['bands'] = lib.nchan()
    if 'wavelength' in meta: del meta['wavelength']
    if 'fwhm' in meta: del meta['fwhm']
    meta['band names'] = lib.groups()
    meta['data type'] = 4
    meta['interleave'] = 'bip'
    D = envi.create_image(depth_outhdr, meta, force=True, ext="")
    P = envi.create_image(post_outhdr,  meta, force=True, ext="")
    L = envi.create_image(nll_outhdr,   meta, force=True, ext="")
    C = envi.create_image(corr_outhdr,    meta, force=True, ext="")

    for r in range(R.shape[0]):

        logging.info('Row %i'%r)
        # We delete the old objects to flush everything to disk, empty cache
        del R; del U; del D; del P; del L; del C
        R = envi.open(rfl_inhdr   )
        U = envi.open(uncert_inhdr)
        D = envi.open(depth_outhdr)
        P = envi.open(post_outhdr )
        L = envi.open(nll_outhdr  )
        C = envi.open(corr_outhdr )
        rmm = R.open_memmap(interleave="source", writable=False)
        umm = U.open_memmap(interleave="source", writable=False)
        dmm = D.open_memmap(interleave="source", writable=True)
        pmm = P.open_memmap(interleave="source", writable=True)
        lmm = L.open_memmap(interleave="source", writable=True)
        cmm = C.open_memmap(interleave="source", writable=True)

        # Get reflectance subframe
        sub_rfl    = sp.array(rmm[r,:,:], dtype='float32')
        if R.metadata['interleave'] == 'bil':
            sub_rfl = sub_rfl.T

        # Get input uncertainty
        sub_uncert = sp.array(umm[r,:,:], dtype='float32')
        if U.metadata['interleave'] == 'bil':
            sub_uncert = sub_uncert.T

        # Set-up for parallel.  By convention, we exclude final state
        # vector uncertainties which are related typically to atmosphere
        nrfl = sub_rfl.shape[1]
        sub_data = sp.stack((sub_rfl, sub_uncert[:, 0:nrfl]), axis=2)

        # Perform the fit
        sub_depth = sp.zeros((sub_rfl.shape[0], lib.nchan()), dtype=sp.float32)
        sub_post  = sp.zeros((sub_rfl.shape[0], lib.nchan()), dtype=sp.float32)
        sub_nll   = sp.zeros((sub_rfl.shape[0], lib.nchan()), dtype=sp.float32)
        sub_corr  = sp.zeros((sub_rfl.shape[0], lib.nchan()), dtype=sp.float32)

        # Fit in parallel
        results = pool.map(lib.fit, [sub_data[c, :, :] for c in range(sub_data.shape[0])])
        results = sp.asarray(results)

        # For debugging, use this instead:
        '''
        results = []
        for c in range(sub_data.shape[0]):
            temp = lib.fit(sub_data[c,:,:])
            results.append(temp)
        results = sp.asarray(results)
        '''
        sub_depth = results[:, 0, :]
        sub_post  = results[:, 1, :]
        sub_nll   = results[:, 2, :]
        sub_corr  = results[:, 3, :]

        # Write to output file
        dmm[r,:,:] = sub_depth
        pmm[r,:,:] = sub_post
        lmm[r,:,:] = sub_nll
        cmm[r,:,:] = sub_corr


if __name__ == "__main__":
    main()
