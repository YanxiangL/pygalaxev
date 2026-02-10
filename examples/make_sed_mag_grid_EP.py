import os
import sys
import numpy as np
import h5py
from scipy.interpolate import splrep, splev, splint, interp1d
from configobj import ConfigObj
from astropy.cosmology import LambdaCDM
from scipy.integrate import simpson
import warnings
# import time


def convert_pardict(pardict):
    """Convert string parameters from ConfigObj to appropriate types.
    Parameters
    ----------
    pardict : dict
        Dictionary of parameters with string values.
    Returns
    -------
    pardict : dict
        Dictionary of parameters with converted values.
    """

    pardict["mu"] = float(pardict["mu"])
    pardict["tau_V"] = float(pardict["tau_V"])
    pardict["epsilon"] = float(pardict["epsilon"])
    pardict["ntau"] = int(pardict["ntau"])
    pardict["nage"] = int(pardict["nage"])
    pardict["tau_grid_lim"] = np.float64(pardict["tau_grid_lim"])
    pardict["age_grid_lim"] = np.float64(pardict["age_grid_lim"])
    pardict["nz_bin"] = int(pardict["nz_bin"])
    pardict["z_grid_lim"] = np.float64(pardict["z_grid_lim"])

    pardict["omega_lambda"] = float(pardict["omega_lambda"])
    pardict["omega_matter"] = float(pardict["omega_matter"])
    pardict["omega_b"] = float(pardict["omega_b"])
    pardict["sigma_8"] = float(pardict["sigma_8"])
    pardict["n_s"] = float(pardict["n_s"])
    pardict["h"] = float(pardict["h"])
    pardict["n_custom_sfh"] = int(pardict["n_custom_sfh"])
    pardict["verbose"] = bool(int(pardict["verbose"]))

    return pardict


def grid_lum_bands(
    grid,
    bands,
    redshift,
    Dlum,
    pardict,
    wave,
    filtdir,
    Z_grid,
    tau_grid,
    age_grid,
    galaxy_id=None,
    index=None,
):
    wave_obs = wave * (1.0 + redshift)  # observed wavelength grid
    if pardict["sfh_model"] == "custom":
        output_file = h5py.File(
            work_dir
            + "/vst_mags_grid_z=%6.4f_sfh_%s_galaxy_%d_fast.hdf5"
            % (redshift, pardict["sfh_model"], galaxy_id),
            "w",
        )
    else:
        output_file = h5py.File(
            work_dir
            + "/vst_mags_grid_z=%6.4f_sfh_%s_fast.hdf5"
            % (redshift, pardict["sfh_model"]),
            "w",
        )
    for band in bands:
        # filtname = filtdir + "%s_OmegaCAM.res" % band
        filtname = filtdir + "/%s_%s.res" % (band, pardict["survey"])

        f = open(filtname, "r")
        filt_wave, filt_t = np.loadtxt(f, unpack=True)
        f.close()

        # filt_spline = splrep(filt_wave, filt_t)
        filt_spline = interp1d(
            filt_wave, filt_t, kind="linear", bounds_error=False, fill_value=0.0
        )

        wmin_filt, wmax_filt = filt_wave[0], filt_wave[-1]
        cond_filt = (wave_obs >= wmin_filt) & (wave_obs <= wmax_filt)
        nu_cond = np.flipud(cond_filt)

        # Evaluate the filter response at the wavelengths of the spectrum
        # response = splev(wave_obs[cond_filt], filt_spline)
        response = filt_spline(wave_obs[cond_filt])
        nu_filter = csol * 1e8 / wave_obs[cond_filt]

        # flips arrays
        response = np.flipud(response)
        nu_filter = np.flipud(nu_filter)

        # filter normalization
        bp = splrep(nu_filter, response / nu_filter, s=0, k=1)
        bandpass = splint(nu_filter[0], nu_filter[-1], bp)

        if pardict["sfh_model"] == "custom":
            mag_grid = np.zeros((nZ, nage))
            lum_grid = np.zeros((nZ, nage, len(nu_filter)))

            for i in range(nZ):
                for k in range(nage):
                    llambda = grid[i, index, k, :]

                    flambda_obs = (
                        llambda
                        * L_Sun
                        / (4.0 * np.pi * (Dlum * Mpc) ** 2)
                        / (1.0 + redshift)
                    )  # observed specific flux in erg/s/cm^2/AA

                    fnu = flambda_obs * wave_obs**2 / csol * 1e-8  # F_nu in cgs units

                    fnu = np.flipud(fnu)
                    # Integrate
                    observed = splrep(
                        nu_filter, response * fnu[nu_cond] / nu_filter, s=0, k=1
                    )
                    flux = splint(nu_filter[0], nu_filter[-1], observed)

                    lum_avg = (
                        fnu[nu_cond]
                        * 4.0
                        * np.pi
                        * (Dlum * Mpc) ** 2
                        * (1.0 + redshift)
                    )  # in erg/s/Hz. The band average luminosity

                    lum_grid[i, k, :] = lum_avg

                    # To get the AB magnitude, we need to minus 48.6 term. To get the mass normalization term, we would need to minus -2.5*log_10_mass, we apply during interpolation.
                    mag_grid[i, k] = -2.5 * np.log10(flux / bandpass) - 48.6

            mag_dset = output_file.create_dataset("%s_mag" % (band), data=mag_grid)

            mag_dset.attrs["units"] = "AB magnitude "
            mag_dset.attrs["axis_0"] = "Metallicity"
            mag_dset.attrs["axis_1"] = "age"

            lum_dset = output_file.create_dataset("%s_lum" % (band), data=lum_grid)
            lum_dset.attrs["units"] = "Luminosity density in erg/s/Hz"
            lum_dset.attrs["axis_0"] = "Metallicity"
            lum_dset.attrs["axis_1"] = "age"

            nu_filter_dset = output_file.create_dataset(
                "%s_nu_filter" % (band), data=nu_filter
            )
            nu_filter_dset.attrs["units"] = "Frequency in Hz"
            nu_filter_dset.attrs["description"] = (
                "Frequencies corresponding to the filter response curve"
            )

            response_dset = output_file.create_dataset(
                "%s_response" % (band), data=response
            )
            response_dset.attrs["units"] = "Dimensionless"
            response_dset.attrs["description"] = (
                "Filter response curve for band %s" % band
            )
        else:
            mag_grid = np.zeros((nZ, ntau, nage))
            lum_grid = np.zeros((nZ, ntau, nage, len(nu_filter)))
            for i in range(nZ):
                for j in range(ntau):
                    for k in range(nage):
                        llambda = grid[i, j, k, :]

                        flambda_obs = (
                            llambda
                            * L_Sun
                            / (4.0 * np.pi * (Dlum * Mpc) ** 2)
                            / (1.0 + redshift)
                        )  # observed specific flux in erg/s/cm^2/AA

                        fnu = (
                            flambda_obs * wave_obs**2 / csol * 1e-8
                        )  # F_nu in cgs units

                        nu_obs = np.flipud(csol / wave_obs * 1e8)
                        fnu = np.flipud(fnu)
                        # Integrate
                        observed = splrep(
                            nu_filter, response * fnu[nu_cond] / nu_filter, s=0, k=1
                        )
                        flux = splint(nu_filter[0], nu_filter[-1], observed)

                        lum_avg = (
                            fnu[nu_cond]
                            * 4.0
                            * np.pi
                            * (Dlum * Mpc) ** 2
                            * (1.0 + redshift)
                        )  # in erg/s/Hz. The band average luminosity

                        lum_grid[i, j, k, :] = lum_avg

                        # To get the AB magnitude, we need to minus 48.6 term. To get the mass normalization term, we would need to minus -2.5*log_10_mass, we apply during interpolation.
                        mag_grid[i, j, k] = -2.5 * np.log10(flux / bandpass) - 48.6

            mag_dset = output_file.create_dataset("%s_mag" % (band), data=mag_grid)

            mag_dset.attrs["units"] = "AB magnitude "
            mag_dset.attrs["axis_0"] = "Metallicity"
            mag_dset.attrs["axis_1"] = "tau"
            mag_dset.attrs["axis_2"] = "age"

            lum_dset = output_file.create_dataset("%s_lum" % (band), data=lum_grid)
            lum_dset.attrs["units"] = "Luminosity density in erg/s/Hz"
            lum_dset.attrs["axis_0"] = "Metallicity"
            lum_dset.attrs["axis_1"] = "tau"
            lum_dset.attrs["axis_2"] = "age"

            nu_filter_dset = output_file.create_dataset(
                "%s_nu_filter" % (band), data=nu_filter
            )
            nu_filter_dset.attrs["units"] = "Frequency in Hz"
            nu_filter_dset.attrs["description"] = (
                "Frequencies corresponding to the filter response curve"
            )

            response_dset = output_file.create_dataset(
                "%s_response" % (band), data=response
            )
            response_dset.attrs["units"] = "Dimensionless"
            response_dset.attrs["description"] = (
                "Filter response curve for band %s" % band
            )

    output_file.create_dataset("Z_grid", data=Z_grid)
    if pardict["sfh_model"] != "custom":
        output_file.create_dataset("tau_grid", data=tau_grid)
    output_file.create_dataset("age_grid", data=age_grid)
    output_file.close()


configfile = sys.argv[1]
pardict = convert_pardict(ConfigObj(configfile))

sys.path.append(pardict["pygalaxev_path"])  # add pygalaxev to the system path
import pygalaxev
import pygalaxev_cosmology
from pygalaxev_cosmology import c as csol, L_Sun, Mpc

nage = pardict["nage"]
tau_V = pardict["tau_V"]
mu = pardict["mu"]
epsilon = pardict["epsilon"]
bands = pardict["bands"]

# creates CSP models on a grid of stellar population parameters using galaxev

# selects the stellar template library:
# Low-resolution 'BaSeL' library, Chabrier IMF
ssp_dir = pardict["ssp_dir"]  # directory containing the SSP .ised files
tempname = pardict["tempname"]  # template name, e.g., 'BaSeL_Chabrier'

work_dir = pardict["work_dir"]  # working directory

# Using Padova 1994 tracks.
Z_given_code = {
    "m22": 0.0001,
    "m32": 0.0004,
    "m42": 0.004,
    "m52": 0.008,
    "m62": 0.02,
    "m72": 0.05,
    "m82": 0.1,
}
nwav = 2023  # size of wavelength grid (can be looked up by running 'csp_galaxev' on any .ised file of the spectral library)

nZ = len(Z_given_code)  # size of metallicity grid
# ntau = 35  # size of grid in exponential timescale
# nage = 31  # size of grid in age

# grid of values of tau, tau_V and age

if pardict["mode_load"] == "local":
    job_id = int(sys.argv[2])
    pardict["out_dir"] = (
        pardict["out_dir"] + f"/job_id_{job_id}_{pardict['galaxy_type']}/"
    )
    pardict["sfh_dir"] = (
        pardict["sfh_dir"] + f"/job_id_{job_id}_{pardict['galaxy_type']}/"
    )
    index_to_hdf5_num_table = np.loadtxt(
        pardict["out_dir"] + "/subhalo_num_per_file.txt"
    )

    num_hdf5_subhalo = int(index_to_hdf5_num_table[-1, -1] + 1)

    tot_job, reminder = divmod(num_hdf5_subhalo, int(pardict["index_per_job"]))

    if reminder > 0 and job_id == tot_job:
        start_index = job_id * int(pardict["index_per_job"])
        end_index = num_hdf5_subhalo
    else:
        start_index = job_id * int(pardict["index_per_job"])
        end_index = (job_id + 1) * int(pardict["index_per_job"])

    galaxy_data = []
    hdf5_num_all = []
    for index in range(start_index, end_index):
        try:
            hdf5_num = np.where(
                (index_to_hdf5_num_table[:, 2] == index)
                & (index_to_hdf5_num_table[:, 1] > 0)
            )[0][0]
            galaxy_data.append(
                np.load(
                    pardict["out_dir"]
                    + "/subhalo_data_kinematics_only_"
                    + str(hdf5_num)
                    + ".npz",
                    allow_pickle=True,
                )["arr_0"]
            )
            hdf5_num_all.append(hdf5_num)
        except FileNotFoundError:
            hdf5_num = np.where(
                (index_to_hdf5_num_table[:, 2] == index)
                & (index_to_hdf5_num_table[:, 1] > 0)
            )[0][0]
            print(
                "Galaxy data file for index "
                + str(hdf5_num)
                + " not found. No galaxy satisfies the selection criteria."
            )
            continue

    if len(galaxy_data) == 0:
        print("No galaxy data found for the given job index. Exiting.")
        sys.exit(0)

else:
    galaxy_data = np.load(
        pardict["out_dir"] + "/subhalo_data_kinematics_only.npz", allow_pickle=True
    )["arr_0"]

out_dir = pardict["out_dir"]

if pardict["sfh_model"] == "custom":
    tau_grid = []
    age_grid_all = []
    redshift_list = []
    galaxy_id_all = []

    if pardict["mode_load"] == "API":
        ntau = pardict["n_custom_sfh"]
        for i in range(ntau):
            tau_grid.append(
                pardict["sfh_dir"]
                + "galaxy_"
                + str(galaxy_data[i]["SubhaloID"])
                + "_sfh.txt"
            )
            age_grid_all.append(
                np.linspace(
                    galaxy_data[i]["min_stellar_age"],
                    galaxy_data[i]["max_stellar_age"],
                    nage,
                )
            )
            redshift_list.append(galaxy_data[i]["Redshift"])
            galaxy_id_all.append(galaxy_data[i]["SubhaloID"])
    else:
        index = 0
        for i in range(len(galaxy_data)):
            for j in range(len(galaxy_data[i])):
                tau_grid.append(
                    pardict["sfh_dir"]
                    + "/galaxy_"
                    + str(galaxy_data[i][j]["SubhaloID"])
                    + "_sfh.txt"
                )
                age_grid_all.append(
                    np.linspace(
                        # np.nextafter(galaxy_data[i][j]["min_stellar_age"], -np.inf),
                        # np.nextafter(galaxy_data[i][j]["max_stellar_age"], np.inf),
                        galaxy_data[i][j]["min_stellar_age"],
                        galaxy_data[i][j]["max_stellar_age"],
                        nage,
                    )
                )
                redshift_list.append(galaxy_data[i][j]["Redshift"])
                galaxy_id_all.append(galaxy_data[i][j]["SubhaloID"])
                index += 1
        ntau = index
        print("Total number of custom SFH models: %d" % ntau)
else:
    ntau = pardict["ntau"]
    tau_grid = np.linspace(pardict["tau_grid_lim"][0], pardict["tau_grid_lim"][1], ntau)

    age_grid = np.linspace(pardict["age_grid_lim"][0], pardict["age_grid_lim"][1], nage)

redshift_list = np.array(redshift_list)
grid = np.zeros((nZ, ntau, nage, nwav))
# index_delete = []
Z_grid = []

for m in range(2, 9):  # loop over metallicities
    Zcode = "m%d2" % m
    Z = Z_given_code[Zcode]
    Z_grid.append(Z)

    for t in range(ntau):
        # Create the models
        isedname = ssp_dir + "/bc2003_%s_%s_chab_ssp.ised" % (tempname, Zcode)
        if pardict["sfh_model"] == "custom":
            cspname = "bc03_Z=%6.4f_tV=%5.3f_mu=%3.1f_eps=%5.3f_galaxy_%d" % (
                Z,
                tau_V,
                mu,
                epsilon,
                galaxy_id_all[t],
            )
        else:
            cspname = "bc03_Z=%6.4f_tau=%5.3f_tV=%5.3f_mu=%3.1f_eps=%5.3f" % (
                Z,
                tau_grid[t],
                tau_V,
                mu,
                epsilon,
            )

        tries = 0
        age_original = np.loadtxt(tau_grid[t])[:, 0].copy()
        tau_original = np.loadtxt(tau_grid[t])[:, 1].copy()

        if pardict["sfh_model"] == "custom":
            nonzero_indices = np.where(tau_original > 0)[0]
            index_start = nonzero_indices[0]

            new_tau = False

            if age_original[index_start] > age_grid_all[t][0] * 1e9:
                print(
                    "Oldest star age not included. This is because sfh is calculated with the histogram. Interpolating to add it."
                )

                sfr_age_min = np.interp(
                    age_grid_all[t][0] * 1e9, age_original, tau_original
                )

                age_original = np.insert(
                    age_original, index_start, age_grid_all[t][0] * 1e9
                )
                tau_original = np.insert(tau_original, index_start, sfr_age_min)

                new_tau = True

                # print(age_grid_all[t][0] * 1e9, sfr_age_min)

            nonzero_indices = np.where(tau_original > 0)[0]
            index_end = nonzero_indices[-1]

            if age_original[index_end] < age_grid_all[t][-1] * 1e9:
                print(
                    "Youngest star age not included. This is because sfh is calculated with the histogram. Interpolating to add it."
                )

                sfr_age_max = np.interp(
                    age_grid_all[t][-1] * 1e9, age_original, tau_original
                )

                age_original = np.insert(
                    age_original, index_end + 1, age_grid_all[t][-1] * 1e9
                )
                tau_original = np.insert(tau_original, index_end + 1, sfr_age_max)

                new_tau = True

            if new_tau:
                # Check age_original is strictly increasing
                if not np.all(np.diff(age_original) > 0):
                    raise ValueError(
                        "age_original is not strictly increasing after adding boundary points. It's a bug."
                        % galaxy_id_all[t]
                    )

                tau_grid[t] = (
                    pardict["sfh_dir"]
                    + "/galaxy_"
                    + str(galaxy_id_all[t])
                    + "_reformated_sfh.txt"
                )
                np.savetxt(
                    tau_grid[t],
                    np.column_stack((age_original, tau_original)),
                )

        while tries < 100:
            try:
                if tries >= 50:
                    verbose_flag = True
                else:
                    verbose_flag = pardict["verbose"]

                pygalaxev.run_csp_galaxev(
                    isedname,
                    cspname,
                    sfh_pars=tau_grid[t],
                    tau_V=tau_V,
                    mu=mu,
                    epsilon=epsilon,
                    work_dir=out_dir,
                    sfh=pardict["sfh_model"],
                    verbose=verbose_flag,
                )

                # Create the mass normalization models
                massname = out_dir + "/" + cspname + ".mass"
                d = np.loadtxt(massname)
                # mass_spline = splrep(
                #     d[:, 0], d[:, 10], k=3, s=0
                # )

                # using the sum of M*_liv+M_rem to renormalize the mass. Interpolate in log space to avoid splines going negative
                age_unique, age_index = np.unique(d[:, 0], return_index=True)
                mass_unique = np.log10(d[age_index, 10])

                mass_spline = splrep(age_unique, mass_unique, k=3, s=0)

                # extracts SEDs on age grid
                oname = out_dir + "/" + cspname + "_agegrid.sed"
                if pardict["sfh_model"] == "custom":
                    age_grid = age_grid_all[t].copy()

                    index_below_min_age = np.where(
                        np.log10(age_grid) + 9.0 < age_unique[1]
                    )[0]
                    if len(index_below_min_age) == 1:
                        if tries == 0:
                            print(
                                f"Setting minimum of the age grid ({age_grid[index_below_min_age][0]}) to the minimum age ({10.0 ** (age_unique[1] - 9.0)}), otherwise galaxev will return sed = 0 for this age."
                            )
                            print(
                                "This is normal as galaxev does not keep the same float precision as input age grid, so the output from galaxev may change slightly."
                            )
                        age_grid[index_below_min_age] = np.nextafter(
                            10.0 ** (age_unique[1] - 9.0), np.inf
                        )

                # Sometimes the age input to galaxevpl is round down and will return sed=0 for that age. If that happens, we will round up the input age.

                tmpname = out_dir + "/tmp.in"
                pygalaxev.create_galaxevpl_config(
                    tmpname,
                    out_dir + "/" + cspname + ".ised",
                    oname,
                    age_grid,
                )

                if verbose_flag:
                    os.system("$bc03/galaxevpl < %s" % tmpname)
                else:
                    os.system("$bc03/galaxevpl < %s > /dev/null 2>&1" % tmpname)

                f = open(oname, "r")
                wsed = np.loadtxt(f)
                f.close()

                if np.max(wsed[:, 1]) <= 0.0:
                    print(
                        "SED has zero values. Possibly due to first age being round down when input to galaxevpl. Galaxevpl will return zero SED when age is slightly below the minimum age in the ised file."
                    )
                    # Remove the previous input file and output file
                    os.system("rm %s" % oname)
                    os.system("rm %s" % (tmpname))
                    pygalaxev.create_galaxevpl_config(
                        tmpname,
                        out_dir + "/" + cspname + ".ised",
                        oname,
                        age_grid,
                        round_up=True,
                    )

                    if verbose_flag:
                        os.system("$bc03/galaxevpl < %s" % tmpname)
                    else:
                        os.system("$bc03/galaxevpl < %s > /dev/null 2>&1" % tmpname)

                    f = open(oname, "r")
                    wsed = np.loadtxt(f)
                    f.close()
                    print("The max sed for the lowest age:", np.max(wsed[:, 1]))

                # The second column corresponds to the lowest age in the age grid

                verbose_flag = False  # reset verbose flag

                print(
                    "The length of SFH input file is %d for galaxy id %d"
                    % (len(np.loadtxt(tau_grid[t])), galaxy_id_all[t])
                )

                if tries > 0:
                    print(
                        "This is a known issue with galaxev. If the ised file has more than 500 lines, it will raise an error."
                    )
                    print(
                        "By pass this problem by reducing the size of the input sfh grid using linear interpolation."
                    )

                break

            except Exception as e:
                tries += 1

                if tries >= 100:
                    raise ValueError(
                        "Error reading file %s after %d tries: %s. The input sfh grid may be too large."
                        % (oname, tries, e)
                    )
                sfh_length = len(tau_original)

                # Find the indices corresponding to the first nonzero sfr and last nonzero sfr.
                nonzero_indices = np.where(tau_original > 0)[0]
                index_start = nonzero_indices[0]
                index_end = nonzero_indices[-1]

                if len(tau_original) > 265:
                    sfh_length_new = 265 - 5 * tries
                else:
                    sfh_length_new = len(tau_original) - 5 * tries

                length_zero = sfh_length_new - (index_end - index_start + 1)

                if length_zero < 0:
                    raise ValueError(
                        "Cannot reduce the size of the input sfh grid further for galaxy id %d"
                        % galaxy_id_all[t]
                    )

                length_zero_end = np.int32(length_zero // 2)
                length_zero_start = length_zero - length_zero_end
                age_grid_new = np.concatenate(
                    (
                        np.linspace(
                            age_original[0],
                            age_original[index_start - 1],
                            length_zero_start,
                        ),
                        age_original[index_start : index_end + 1],
                        np.linspace(
                            age_original[index_end + 1],
                            age_original[-1],
                            length_zero_end,
                        ),
                    )
                )
                tau_grid_new = np.concatenate(
                    (
                        np.zeros(length_zero_start),
                        tau_original[index_start : index_end + 1],
                        np.zeros(length_zero_end),
                    )
                )

                # sfh_length_new = sfh_length - 5 * tries

                # age_grid_new = np.linspace(
                #     age_original[0], age_original[-1], sfh_length_new
                # )

                # tau_grid_new = np.interp(age_grid_new, age_original, tau_original)

                tau_grid[t] = (
                    pardict["sfh_dir"]
                    + "/galaxy_"
                    + str(galaxy_id_all[t])
                    + "_tries_"
                    + str(tries)
                    + "_sfh.txt"
                )
                np.savetxt(tau_grid[t], np.column_stack((age_grid_new, tau_grid_new)))
                # raise ValueError("Error reading file %s: %s" % (oname, e))
                # # Put a placeholder for handling the error
                # for a in range(nage):
                #     grid[m - 2, t, a, :] = None
                # index_delete.append((t))
                # continue

        wave = wsed[:, 0]

        for a in range(nage):
            flux = wsed[:, a + 1]

            # Renormalize the mass!
            if pardict["sfh_model"] == "custom":
                age = age_grid_all[t][a]
            else:
                age = age_grid[a]

            logAge = np.log10(age) + 9.0  # Convert age in Gyr to logAge in yr
            if logAge < age_unique[1] and logAge >= age_unique[0]:
                # # use linear extrapolation for ages below the minimum age in the mass_spline
                # mass = 10 ** (
                #     mass_unique[0]
                #     + (mass_unique[1] - mass_unique[0])
                #     / (age_unique[1] - age_unique[0])
                #     * (logAge - age_unique[0])
                # )

                # Setting it to the minimum age from galaxev. It assumes all ages below this are the same.
                mass = 10 ** mass_unique[1]
            # Avoid extrapolation on both ends
            elif logAge < age_unique[0] or logAge > age_unique[-1]:
                raise ValueError(
                    "Age %f out of bounds for mass normalization spline." % (age)
                )
            else:
                mass = 10 ** splev(logAge, mass_spline)
            sed = flux / mass

            grid[m - 2, t, a, :] = sed

            if np.max(sed) <= 0.0:
                raise ValueError(
                    "Negative or zero SED values found for Z=%6.4f, tau index=%d, age index=%d, galaxy id=%d, sed min=%e, sed max=%e, age=%f"
                    % (Z, t, a, galaxy_id_all[t], np.min(sed), np.max(sed), age)
                )

        # Clean up
        os.system("rm %s" % oname)

        if pardict["sfh_model"] == "custom":
            print("Done Z=%6.4f, galaxy id=%d" % (Z, galaxy_id_all[t]))
            # Delete the intermediate files to save space. For some reason, galaxev output these files to pygalaxev_path instead of work_dir
            os.system("rm -f %s/junk*" % (pardict["pygalaxev_path"]))
        else:
            print("Done Z=%6.4f, tau=%5.3f" % (Z, tau_grid[t]))

# if len(index_delete) > 0:
#     print("Warning: Some files could not be read and were skipped:", index_delete)
#     grid = np.delete(grid, index_delete, axis=1)
#     redshift_list = np.delete(redshift_list, index_delete, axis=0)
#     tau_grid = np.delete(tau_grid, index_delete, axis=0)
#     age_grid_all = np.delete(age_grid_all, index_delete, axis=0)
#     galaxy_id_all = np.delete(galaxy_id_all, index_delete, axis=0)

Z_grid = np.array(Z_grid)

# # Save the SED grid
# grid_file = h5py.File(
#     work_dir
#     + "/BaSeL_Chabrier_sed_grid_"
#     + str(mu)
#     + "_"
#     + str(tau_V)
#     + "_"
#     + str(epsilon)
#     + "_"
#     + str(pardict["sfh_model"])
#     + ".hdf5",
#     "w",
# )
# grid_dset = grid_file.create_dataset("sed_grid", data=grid)
# grid_dset.attrs["units"] = (
#     "Llambda (in units of L_Sun/Angstrom) for 1M_Sun (living + remnants)"
# )
# grid_dset.attrs["axis_0"] = "Metallicity"
# if pardict["sfh_model"] == "custom":
#     grid_dset.attrs["axis_1"] = "galaxy id"
# else:
#     grid_dset.attrs["axis_1"] = "tau"
# grid_dset.attrs["axis_2"] = "age"
# grid_dset.attrs["axis_3"] = "Wavelength"

# grid_file.create_dataset("Z_grid", data=Z_grid)
# if pardict["sfh_model"] == "custom":
#     grid_file.create_dataset("tau_grid", data=np.arange(ntau))
# else:
#     grid_file.create_dataset("tau_grid", data=tau_grid)
# grid_file.create_dataset("age_grid", data=age_grid)
# grid_file.create_dataset("wave", data=wave)

# ===============================================================================
# Now compute magnitudes on a grid of redshifts
# ===============================================================================
# filtdir = os.environ.get("PYGALAXEVDIR") + "/filters/"
filtdir = pardict["filtdir"]  # directory containing filter response curves

cosmo = LambdaCDM(
    H0=pardict["h"] * 100,
    Om0=pardict["omega_matter"],
    Ob0=pardict["omega_b"],
    Ode0=pardict["omega_lambda"],
)  # define cosmology

if pardict["sfh_model"] == "custom":
    Dlum = cosmo.luminosity_distance(redshift_list).value  # luminosity distance in Mpc
    for i in range(len(redshift_list)):
        grid_lum_bands(
            grid,
            bands,
            redshift_list[i],
            Dlum[i],
            pardict,
            wave,
            filtdir,
            Z_grid,
            tau_grid,
            age_grid_all[i],
            galaxy_id=galaxy_id_all[i],
            index=i,
        )

else:
    redshift = np.linspace(
        pardict["z_grid_lim"][0], pardict["z_grid_lim"][1], pardict["nz_bin"]
    )  # redshift grid
    Dlum = cosmo.luminosity_distance(redshift).value  # luminosity distance in Mpc

    for z_red in range(len(redshift)):
        grid_lum_bands(
            grid,
            bands,
            redshift[z_red],
            Dlum[z_red],
            pardict,
            wave,
            filtdir,
            Z_grid,
            tau_grid,
            age_grid,
            galaxy_id=galaxy_id_all[i],
        )
