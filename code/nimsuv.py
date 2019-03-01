import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.convolution import discretize_model
from astropy.modeling.models import Gaussian2D
from astro.table import Table
from photutils.datasets import (make_gaussian_sources_image,
                                make_noise_image)
from tqdm import tqdm, trange


__all__ = ['ImageSimulation']


class ImageSimulation(object):
    """Simulate images for the nimble spacecraft SUVOIR instrument

    Parameters
    ----------
    bandpass_method: str
        The method used to calculate the bandpass when converting from
        flux to photons

        * "simple": provide only a central wavelength and bandpass width

    """

    def __init__(self, texp=600, mirror_diameter=0.3, throughput=0.8,
                 image_shape=(100, 100), arcsec_per_pixel=4,
                 QE=0.9, psf_model='Gaussian', psf_oversample=20,
                 psf_shape=(3, 3),
                 bandpass_method='simple', lam=None, dlam=None,
                 jitter=0., texp_sub=None):
        self.texp = texp * u.s
        self.texp_sub = texp_sub
        if texp_sub is not None:
            self.texp_sub *= u.s
        self.mirror_diameter_m = mirror_diameter * u.m
        self.throughput = throughput
        self.QE = QE * u.ct / u.photon
        self.image_shape_pix = image_shape
        self.arcsec_per_pix = arcsec_per_pixel * u.arcsec
        self.imagearea_arcsec2 = (self.image_shape[0] *
                                  self.image_shape[1] *
                                  self.arcsec_per_pix**2)
        self.psf_model = psf_model
        if psf_model is not 'Gaussian':
            raise NotImplementedError("Only Gaussian PSF profiles are "
                                      "implemented")
        self.psf_oversample = psf_oversample
        self.psf_shape_arcsec = psf_shape * u.arcsec
        self.bandpass_method = bandpass_method
        if self.bandpass_method is not 'simple':
            raise NotImplementedError("Only simple bandpasses are implemented")
        self.lam = lam * u.um
        self.dlam = dlam * u.um
        if (self.bandpass_method is 'simple') and (self.lam is None):
            raise ValueError("If bandpass_method is simple, then lam and "
                             "dlam must be set")
        if (self.bandpass_method is 'simple') and (self.dlam is None):
            raise ValueError("If bandpass_method is simple, then lam and "
                             "dlam must be set")
        self.jitter = jitter
        if self.jitter > 0:
            raise NotImplementedError("Jitter is not yet implemented")

    def make_background_stars_image(self, bg_star_magnitudes, noisey=True):
        """ make a noiseless image of background stars
        takes a list of background star fluxes and randomly
        places stars in your image.
        """
        # convert to Jy (10−23 erg s−1 Hz−1 cm−2)
        bgstarsflux = 10**((bg_star_magnitudes - 8.9) / -2.5) * u.Jansky
        bgstars_photons = (bgstarsflux.to(u.photon / u.cm**2 / u.s / u.um,
                                          equivalencies=u.spectral_density(
                                              self.lam * u.um)) *
                           self.mirror_area.to(u.cm**2) * self.dlam)
        

    def make_reference_image(self, noisey=True):
        pass

    def make_background_noise_image(self):
        pass

    def make_target_image(self, magntiude=21, noisey=True):
        pass

    def make_host_image(self, magnitude='default', noisey=True):
        pass
