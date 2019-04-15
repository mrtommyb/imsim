import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.convolution import discretize_model
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
from photutils.datasets import (make_gaussian_sources_image,
                                make_noise_image)
from tqdm import tqdm, trange
from multiprocessing import Pool

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

    def __init__(self, texp=60, mirror_diameter=0.3, throughput=0.8,
                 image_shape=[100, 100], arcsec_per_pixel=4,
                 QE=0.9, psf_model='Gaussian', psf_oversample=20,
                 psf_shape=(3, 3, 0), ncoadds=1,
                 bandpass_method='simple', lam=None, dlam=None,
                 jitter=0.,
                 zodi=4.e-6, black=1, rstate=None,
                 bg_star_magnitudes=None,
                 bg_star_area=3.14159,
                 host_mag=None, target_mag=21, host_shape=(10, 5, 0),
                 target_sepatation_function='Gaussian',
                 target_sepatation=10, parallel=False):
        if rstate is not None:
            np.random.seed(rstate)
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
        self.psf_shape_arcsec = psf_shape[0:2] * u.arcsec
        self.psf_shape_theta = psf_shape[2] * u.radian
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
        self.zodi_per_arcsec = zodi * u.Jy
        _zodi_per_pix_ps = (self.zodi_per_arcsec *
                            self.arcsec_per_pix.value**2 * c.c /
                            self.lam**2 * self.dlam *
                            self.mirror_area_m2 /
                            (c.h * c.c / self.lam / u.photon)
                            ).to(u.photon / u.s)
        self.zodi_ct_per_pix = _zodi_per_pix_ps * self.throughput * self.QE
        self.black_ct_per_pix = black

        _ = self._set_background_star_positions(
            bg_star_magnitudes=bg_star_magnitudes,
            bg_star_area=bg_star_area,
            rstate=rstate)

        _ = self._set_targethost_positions(
            target_mag=target_mag,
            host_mag=host_mag,
            target_sepatation=target_sepatation,
            target_sepatation_function=target_sepatation_function,
            host_shape=host_shape,
            rstate=rstate)
        self.parallel = parallel

    def _set_background_star_positions(self, bg_star_magnitudes,
                                       bg_star_area=3.14159,
                                       rstate=None):
        if rstate is not None:
            np.random.seed(rstate)
        table_bgstars = Table()
        if bg_star_magnitudes is None:
            self.table_bgstars = table_bgstars
            self.nstars_in_image = 0
            return
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
        table_bgstars['flux'] = fval

        x_mean = np.random.uniform(low=2,
            high=self.image_shape_pix[0] - 2, size=self.nstars_in_image)
        y_mean = np.random.uniform(low=2,
            high=self.image_shape_pix[1] - 2, size=self.nstars_in_image)
        table_bgstars['x_mean'] = x_mean
        table_bgstars['y_mean'] = y_mean

        table_bgstars['x_stddev'] = ((np.zeros_like(
            table_bgstars['x_mean']) + self.psf_shape_arcsec[0].value) /
            self.arcsec_per_pix.value)
        table_bgstars['y_stddev'] = ((np.zeros_like(
            table_bgstars['y_mean']) + self.psf_shape_arcsec[0].value) /
            self.arcsec_per_pix.value)
        table_bgstars['theta'] = (np.zeros(self.nstars_in_image) +
                                  self.psf_shape_theta)

        self.table_bgstars = table_bgstars

    def _set_targethost_positions(self,
                                  target_mag, host_mag,
                                  host_shape,
                                  target_sepatation,
                                  target_sepatation_function,
                                  rstate=None):

        if rstate is not None:
            np.random.seed(rstate)

        if host_mag is None:
            host_mag = target_mag - 5

        flux_host = 10**((host_mag - 8.9) / -2.5) * u.Jansky
        flux_target = 10**((target_mag - 8.9) / -2.5) * u.Jansky

        photons_host = (flux_host.to(u.photon / u.cm**2 / u.s / u.um,
                                     equivalencies=u.spectral_density(self.lam)) *
                        self.mirror_area_m2.to(u.cm**2) * self.dlam)
        cnts_host = photons_host * self.throughput * self.QE * self.texp

        photons_target = (flux_target.to(u.photon / u.cm**2 / u.s / u.um,
                                         equivalencies=u.spectral_density(self.lam)) *
                          self.mirror_area_m2.to(u.cm**2) * self.dlam)
        cnts_target = photons_target * self.throughput * self.QE * self.texp

        # we don't put targets right next to the edge of the fov
        x_mean = np.random.uniform(low=2,
            high=self.image_shape_pix[0] - 2)
        y_mean = np.random.uniform(low=2,
            high=self.image_shape_pix[1] - 2)

        if target_sepatation_function == 'Gaussian':
            x_targ, y_targ = np.random.multivariate_normal(
                np.array([x_mean, y_mean]),
                cov=np.array([[target_sepatation, 0], [0, target_sepatation]]))
        else:
            raise NotImplementedError(
                "Only Gaussian target location implemented")

        table_target = Table()
        table_target['flux'] = [cnts_target.value, cnts_host.value]
        table_target['x_mean'] = [x_targ, x_mean]
        table_target['y_mean'] = [y_targ, y_mean]
        table_target['x_stddev'] = [self.psf_shape_arcsec[0].value /
                                    self.arcsec_per_pix.value, host_shape[0] /
                                    self.arcsec_per_pix.value]
        table_target['y_stddev'] = [self.psf_shape_arcsec[0].value /
                                    self.arcsec_per_pix.value, host_shape[1] /
                                    self.arcsec_per_pix.value]
        table_target['theta'] = [self.psf_shape_theta.value,
                                 host_shape[2]]

        self.table_targets = table_target

    def make_science_image(self, rstate=None):

        if rstate is not None:
            np.random.seed(rstate)

        science_image = np.zeros(self.image_shape_pix)
        if self.parallel:
            pool = Pool(8)
            coadds = pool.map(self._science_image_loop,
                                         trange(self.ncoadds))
            return np.array(list(coadds)).sum(axis=0)
        else:
            coadds = np.array(list(map(self._science_image_loop,
                                       trange(self.ncoadds))))
            return science_image + coadds.sum(axis=0)
        # for i in trange(self.ncoadds):
        #     jitter_x = np.random.normal(loc=0, scale=self.jitter_pix)
        #     jitter_y = np.random.normal(loc=0, scale=self.jitter_pix)
        #     bg_star_image = self._make_bg_star_image(noisy=True,
        #                                              jitter=(jitter_x, jitter_y))
        #     target_image = self._make_target_image(noisy=True,
        #                                            jitter=(jitter_x, jitter_y))
        #     host_image = self._make_host_image(noisy=True,
        #                                        jitter=(jitter_x, jitter_y))
        #     noise_image = self._make_noise_image(noisy=True)
        #     science_image += (bg_star_image + noise_image + target_image +
        #                       host_image)
        # return science_image

    def _science_image_loop(self, i):
        jitter_x = np.random.normal(loc=0, scale=self.jitter_pix)
        jitter_y = np.random.normal(loc=0, scale=self.jitter_pix)
        bg_star_image = self._make_bg_star_image(noisy=True,
                                                 jitter=(jitter_x, jitter_y))
        target_image = self._make_target_image(noisy=True,
                                               jitter=(jitter_x, jitter_y))
        host_image = self._make_host_image(noisy=True,
                                           jitter=(jitter_x, jitter_y))
        noise_image = self._make_noise_image(noisy=True)
        science_image_frame = (bg_star_image + noise_image + target_image +
                          host_image)
        return science_image_frame

    def make_reference_image(self):
        ref_image = np.zeros(self.image_shape_pix)
        bg_star_image = self._make_bg_star_image(
            noisy=False, coadd=True, jitter=None)
        host_image = self._make_host_image(
            noisy=False, coadd=True, jitter=None)
        noise_image = self._make_noise_image(noisy=False, coadd=True)
        ref_image += (bg_star_image + noise_image + host_image)
        return ref_image

    def _make_bg_star_image(self, noisy=True, jitter=None, coadd=False):
        table_bgstars_image = Table()
        if self.nstars_in_image == 0:
            return np.zeros(self.image_shape_pix)

        if coadd:
            fval = np.copy(self.table_bgstars['flux']) * self.ncoadds
        else:
            fval = np.copy(self.table_bgstars['flux'])

        x_mean_orig = np.copy(self.table_bgstars['x_mean'])
        y_mean_orig = np.copy(self.table_bgstars['y_mean'])
        if noisy:
            table_bgstars_image['flux'] = np.random.poisson(fval)
        else:
            table_bgstars_image['flux'] = fval

        if jitter is not None:
            jitter_x = jitter[0]
            jitter_y = jitter[1]
            table_bgstars_image['x_mean'] = x_mean_orig + jitter_x  # * u.pixel
            table_bgstars_image['y_mean'] = y_mean_orig + jitter_y  # * u.pixel
        else:
            table_bgstars_image['x_mean'] = x_mean_orig
            table_bgstars_image['y_mean'] = y_mean_orig

        table_bgstars_image['x_stddev'] = np.copy(self.table_bgstars['x_stddev'])
        table_bgstars_image['y_stddev'] = np.copy(self.table_bgstars['y_stddev'])
        table_bgstars_image['theta'] = np.copy(self.table_bgstars['theta'])

        image_bgstars = self._make_background_sources_image(
            table_bgstars_image)
        return image_bgstars

    def _make_background_sources_image(self, table_bgstars, nsigma=6):
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
            theta = table_bgstars['theta'][i]
            image = self.make_gaussian_source_image(image,
                                                    amplitude, x_mean, y_mean,
                                                    x_stddev, y_stddev,
                                                    theta, nsigma)
        return image

    def make_gaussian_source_image(self, image, amplitude, x_mean, y_mean,
                                   x_stddev, y_stddev,
                                   theta, nsigma):
        gmodel = Gaussian2D(amplitude=amplitude,
                            x_mean=0,
                            y_mean=0,
                            x_stddev=x_stddev,
                            y_stddev=y_stddev,
                            theta=theta)
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

    def _make_noise_image(self, noisy=True, coadd=False):

        if coadd:
            texp = self.texp.value * self.ncoadds
        else:
            texp = self.texp.value

        if noisy:
            image_zodi = make_noise_image(
                self.image_shape_pix, type='poisson',
                mean=self.zodi_ct_per_pix.value *
                texp)
            image_black = make_noise_image(
                self.image_shape_pix, type='poisson',
                mean=self.black_ct_per_pix *
                texp)
        else:
            image_zodi = (np.zeros(self.image_shape_pix) +
                          self.zodi_ct_per_pix.value * texp)
            image_black = (np.zeros(self.image_shape_pix) +
                           self.black_ct_per_pix * texp)
        return image_zodi + image_black

    def _make_target_image(self, noisy=True, coadd=False, jitter=None):
        image_target = np.zeros(self.image_shape_pix)
        if coadd:
            fval = self.table_targets['flux'][0] * self.ncoadds
        else:
            fval = self.table_targets['flux'][0]

        if noisy:
            fval = np.random.poisson(fval)

        amplitude = fval / (2. * np.pi *
                            self.table_targets['x_stddev'][0] *
                            self.table_targets['y_stddev'][0])
        if jitter is not None:
            jitter_x = jitter[0]
            jitter_y = jitter[1]
            x_mean = self.table_targets['x_mean'][0] + jitter_x
            y_mean = self.table_targets['y_mean'][0] + jitter_y
        else:
            x_mean = self.table_targets['x_mean'][0]
            y_mean = self.table_targets['y_mean'][0]
        x_stddev = self.table_targets['x_stddev'][0]
        y_stddev = self.table_targets['y_stddev'][0]
        theta = self.table_targets['theta'][0]
        image_target = self.make_gaussian_source_image(image_target,
                                                       amplitude, x_mean, y_mean, x_stddev, y_stddev,
                                                       theta, nsigma=6)
        return image_target

    def _make_host_image(self, noisy=True, coadd=False, jitter=None):
        image_host = np.zeros(self.image_shape_pix)
        if coadd:
            fval = self.table_targets['flux'][1] * self.ncoadds
        else:
            fval = self.table_targets['flux'][1]

        if noisy:
            fval = np.random.poisson(fval)

        amplitude = fval / (2. * np.pi *
                            self.table_targets['x_stddev'][1] *
                            self.table_targets['y_stddev'][1])
        if jitter is not None:
            jitter_x = jitter[0]
            jitter_y = jitter[1]
            x_mean = self.table_targets['x_mean'][1] + jitter_x
            y_mean = self.table_targets['y_mean'][1] + jitter_y
        else:
            x_mean = self.table_targets['x_mean'][1]
            y_mean = self.table_targets['y_mean'][1]
        x_stddev = self.table_targets['x_stddev'][1]
        y_stddev = self.table_targets['y_stddev'][1]
        theta = self.table_targets['theta'][1]
        image_host = self.make_gaussian_source_image(image_host,
                                                     amplitude, x_mean, y_mean, x_stddev, y_stddev,
                                                     theta, nsigma=6)
        return image_host
