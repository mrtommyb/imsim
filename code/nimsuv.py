import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.convolution import discretize_model
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
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
                 image_shape=[100, 100], arcsec_per_pixel=4,
                 QE=0.9, psf_model='Gaussian', psf_oversample=20,
                 psf_shape=(3, 3), ncoadds=1,
                 bandpass_method='simple', lam=None, dlam=None,
                 jitter=0.):
        self.texp = texp * u.s
        self.ncoadds = ncoadds
        self.mirror_diameter_m = mirror_diameter * u.m
        self.mirror_area_m2 = self.mirror_diameter_m**2 * np.pi
        self.throughput = throughput
        self.QE = QE * u.ct / u.photon
        self.image_shape_pix = list(image_shape)
        self.arcsec_per_pix = arcsec_per_pixel * u.arcsec
        self.imagearea_arcsec2 = (self.image_shape_pix[0] *
                                  self.image_shape_pix[1] *
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
        self.jitter_arcsec = jitter * u.arcsec
        self.jitter_pix = self.jitter_arcsec / self.arcsec_per_pix

    def make_background_stars_image(self, bg_star_magnitudes,
                                    bg_star_area=3.14159, noisy=True,
                                    rstate=None):
        """ make a noiseless image of background stars
        takes a list of background star fluxes and randomly
        places stars in your image.
        """
        # convert to Jy (10−23 erg s−1 Hz−1 cm−2)
        if rstate is not None:
            np.random.seed(rstate)
        _nstars_per_square_degree = int(
            len(bg_star_magnitudes) // bg_star_area)
        nstars_in_image = self.imagearea_arcsec2.to(
            u.deg**2) * _nstars_per_square_degree
        self.nstars_in_image = np.int(nstars_in_image.value)

        bgstarsflux = 10**((bg_star_magnitudes - 8.9) / -2.5) * u.Jansky
        bgstars_photons = (bgstarsflux.to(u.photon / u.cm**2 / u.s / u.um,
                                          equivalencies=u.spectral_density(
                                              self.lam)) *
                           self.mirror_area_m2.to(u.cm**2) * self.dlam)
        bgstars_cnts = bgstars_photons * self.throughput * self.QE * self.texp
        table_bgstars = Table()
        fval = np.random.choice(bgstars_cnts, self.nstars_in_image)

        if noisy:
            table_bgstars['flux'] = np.random.poisson(fval)
        else:
            _ = np.random.poisson(fval)  # needed to get the same random state
            table_bgstars['flux'] = fval
        jitter_x = np.random.normal(loc=0, scale=self.jitter_pix,
            size=self.nstars_in_image)
        jitter_y = np.random.normal(loc=0, scale=self.jitter_pix,
            size=self.nstars_in_image)
        table_bgstars['x_mean'] = np.random.uniform(
            high=self.image_shape_pix[0], size=self.nstars_in_image) + jitter_x  # * u.pixel
        table_bgstars['y_mean'] = np.random.uniform(
            high=self.image_shape_pix[1], size=self.nstars_in_image) + jitter_y  # * u.pixel
        table_bgstars['x_stddev'] = ((np.zeros_like(
            table_bgstars['x_mean']) + self.psf_shape_arcsec[0].value) /
            self.arcsec_per_pix.value)
        table_bgstars['y_stddev'] = ((np.zeros_like(
            table_bgstars['y_mean']) + self.psf_shape_arcsec[0].value) /
            self.arcsec_per_pix.value)
        table_bgstars['theta'] = np.zeros(self.nstars_in_image) * u.radian

        # we should use this code in a unit test to check results
        # it's very slow
        # image_bgstars = make_gaussian_sources_image(self.image_shape_pix,
        #                                             table_bgstars,
        #                                             oversample=self.psf_oversample)

        image_bgstars = self._make_gaussian_sources_image(table_bgstars)

        if self.ncoadds > 1:
            for i in range(1,self.ncoadds):
                if noisy:
                    table_bgstars['flux'] = np.random.poisson(fval)
                else:
                    _ = np.random.poisson(fval)  # needed to get the same random state
                jitter_x = np.random.normal(loc=0, scale=self.jitter_pix,
                    size=self.nstars_in_image)
                jitter_y = np.random.normal(loc=0, scale=self.jitter_pix,
                    size=self.nstars_in_image)
                table_bgstars['x_mean'] = np.random.uniform(
                    high=self.image_shape_pix[0], size=self.nstars_in_image) + jitter_x  # * u.pixel
                table_bgstars['y_mean'] = np.random.uniform(
                    high=self.image_shape_pix[1], size=self.nstars_in_image) + jitter_y  # * u.pixel
                image_bgstars += self._make_gaussian_sources_image(table_bgstars)

            
        return image_bgstars

    def _make_gaussian_sources_image(self, table_bgstars, nsigma=6):
        "the one from photutils was too slow"
        image = np.zeros(self.image_shape_pix)
        for i in range(self.nstars_in_image):
            amplitude = table_bgstars['flux'][i] / (2. * np.pi *
                                                    table_bgstars['x_stddev'][i] *
                                                    table_bgstars['y_stddev'][i])
            x_mean = table_bgstars['x_mean'][i]
            y_mean = table_bgstars['y_mean'][i]
            x_stddev = table_bgstars['x_stddev'][i]
            y_stddev = table_bgstars['y_stddev'][i]
            gmodel = Gaussian2D(amplitude=amplitude,
                                x_mean=0,
                                y_mean=0,
                                x_stddev=x_stddev,
                                y_stddev=y_stddev,
                                theta=table_bgstars['theta'][i])
            x_range = self._get_range(
                mean_val=x_mean, sigma=x_stddev, nsigma=nsigma, axis='x')
            y_range = self._get_range(
                mean_val=y_mean, sigma=y_stddev, nsigma=nsigma, axis='y')
            dmodel = discretize_model(model=gmodel,
                                      x_range=tuple(x_range - x_mean),
                                      y_range=tuple(y_range - y_mean),
                                      mode='oversample',
                                      factor=self.psf_oversample)
            image[int(x_range[0]):int(x_range[1]),
                  int(y_range[0]):int(y_range[1])] += dmodel.T
        return image

    def _get_range(self, mean_val, sigma, nsigma, axis='x'):
        if axis == 'x':
            i = 0
        elif axis == 'y':
            i = 1
        v_range = (np.array([-sigma, sigma]) * nsigma) + mean_val
        # v_range *= nsigma
        # v_range += mean_val
        if v_range[0] < 0:
            v_range[0] = 0
        if v_range[1] > self.image_shape_pix[i]:
            v_range[1] = self.image_shape_pix[i]
        v_range[0] = np.floor(v_range[0])
        v_range[1] = np.ceil(v_range[1])
        return v_range

    def make_reference_image(self, noisy=True):
        pass

    def make_background_noise_image(self):
        pass

    def make_target_image(self, magntiude=21, noisy=True):
        pass

    def make_host_image(self, magnitude='default', noisy=True):
        pass
