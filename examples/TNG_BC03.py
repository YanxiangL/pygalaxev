import numpy as np
import h5py
import glob
from configobj import ConfigObj
import tarfile
import gzip
import zipfile
import io
from scipy.io import FortranFile
import sys


def load_compressed_bc_file(filepath, expected_z):
    """Extracts text from a compressed file into RAM and loads it as a NumPy array."""
    content = ""

    # 1. Handle Tarballs (.tar, .tar.gz)
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, "r:*") as tar:
            # Get the first file inside the archive
            member = tar.getmembers()[0]
            content = tar.extractfile(member).read().decode("utf-8")

    # 2. Handle Zip files (.zip)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, "r") as z:
            # Get a list of everything inside the zip
            archive_contents = z.namelist()

            # Find the first actual file (ignoring structural directory entries that end in '/')
            target_file = [name for name in archive_contents if not name.endswith("/")][
                0
            ]

            # Extract and decode the text
            content = z.read(target_file).decode("utf-8")

    # 3. Handle standard Gzip (.gz)
    else:
        try:
            with gzip.open(filepath, "rt") as f:
                content = f.read()
        except OSError:
            # 4. Fallback: If it's not compressed at all, read as plain text
            with open(filepath, "r") as f:
                content = f.read()

    # Verify the metallicity header before parsing the numbers
    assert "Z=%g" % expected_z in content or f"Z={expected_z:g}" in content.replace(
        " ", ""
    ), f"Error: Expected Metallicity {expected_z} not found in header of {filepath}"

    # Convert the raw text string directly into a numpy array
    return np.loadtxt(io.StringIO(content))


def load_binary_ised(filepath):
    """Directly parses the raw BC03 Fortran binary .ised file!"""
    with FortranFile(filepath, "r") as f:
        # 1. First record: Metadata + Ages
        rec0 = f.read_record(np.uint8)
        nsteps = np.frombuffer(rec0[:4], dtype=np.int32)[0]
        ages = np.frombuffer(rec0[4 : 4 + 4 * nsteps], dtype=np.float64)

        # 2. Second record: Wavelengths
        rec1 = f.read_record(np.uint8)
        nwaves = np.frombuffer(rec1[:4], dtype=np.int32)[0]
        wavelengths = np.frombuffer(rec1[4 : 4 + 4 * nwaves], dtype=np.float64)

        # 3. Next nsteps records: The full SED arrays
        seds = np.zeros((nsteps, nwaves), dtype=np.float64)
        for i in range(nsteps):
            rec_flux = f.read_record(np.uint8)
            seds[i, :] = np.frombuffer(rec_flux[4 : 4 + 4 * nwaves], dtype=np.float64)

    # The .ised binary starts at Age=0 (221 steps).
    # The .1ABmag files start at Age=10^5 yr (220 steps).
    # We slice off the 0-age row to perfectly align them!
    return wavelengths, seds[1:, :], ages[1:]


def makeStellarPhotometricsHDF5_BC03(pardict):
    """Create stellar_photometrics.hdf5 file using BC03 models, as used for Illustris and IllustrisTNG runs.
    Bands: UBVK (Buser U,B3,V,IR K filter + Palomar200 IR detectors + atmosphere.57) in Vega, griz (sdss) in AB
    Requires: http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_1994_chabrier_imf.tar.gz
    Produces: 87f665fe5cdac109b229973a2b48f848  stellar_photometrics.hdf5
    Original: f4bcd628b35036f346b4e47f4997d55e  stellar_photometrics.hdf5
      (all datasets between the two satisfy np.allclose(rtol=1e-8,atol=8e-4))
    """

    # filenames1 = sorted(glob.glob("bc2003_hr_m*_chab_ssp.1color"))  # m22-m72
    # filenames2 = sorted(glob.glob("bc2003_hr_m*_chab_ssp.1ABmag"))  # m22-m72

    filenames1 = sorted(
        glob.glob(
            pardict["ssp_dir"]
            + "bc2003_"
            + pardict["tempname"]
            + "_m*_chab_ssp.1ABmag*"
        )
    )  # m22-m82
    # filenames2 = sorted(
    #     glob.glob(
    #         pardict["ssp_dir"]
    #         + "bc2003_"
    #         + pardict["tempname"]
    #         + "_m*_chab_ssp.1color"
    #     )
    # )  # m22-m82

    pardict["tempname"]

    filenames_sed = sorted(
        glob.glob(
            pardict["ssp_dir"] + "bc2003_" + pardict["tempname"] + "_m*_chab_ssp.ised"
        )
    )

    # linear metallicities (mass_metals/mass_total), not in solar!
    Zvals = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1]
    if pardict["survey"] != "SDSS":
        raise ValueError(
            "Currently only SDSS survey is supported, please modify the code if you want to use a different survey."
        )
    # bandNames = ["U", "B", "V", "K", "g", "r", "i", "z"]
    bandNames = pardict[
        "bands"
    ]  # only SDSS bands, since we are only using the ABmag files

    nAgeVals = 220
    # assert len(Zvals) == len(filenames1) == len(filenames2)
    assert len(Zvals) == len(filenames1) == len(filenames_sed), (
        f"Error: Expected {len(Zvals)} metallicity files, but found {len(filenames1)} 1ABmag files and {len(filenames_sed)} .ised files. Please check the filenames and paths."
    )

    # allocate
    ages = np.zeros(nAgeVals)
    mags = {}
    for bandName in bandNames:
        mags[bandName] = np.zeros([len(Zvals), nAgeVals])

    wavelengths = None
    seds_grid = None

    # load BC03 model files
    for i in range(len(Zvals)):
        # data1 = np.loadtxt(filenames1[i])
        # data2 = np.loadtxt(filenames2[i])

        data1 = load_compressed_bc_file(filenames1[i], Zvals[i])
        # data2 = load_compressed_bc_file(filenames2[i], Zvals[i])

        # verify expected number of rows/ages, and that we process the correct metallicity files
        # assert data1.shape[0] == data2.shape[0] == nAgeVals
        # assert data1.shape[0] == nAgeVals
        # with open(filenames1[i], "r") as f:
        #     assert "Z=%g" % Zvals[i] in f.read()
        # with open(filenames2[i], "r") as f:
        #     assert "Z=%g" % Zvals[i] in f.read()

        # ages = data1[:, 0] - 9.0  # log yr -> log Gyr, same in all files
        # mags["U"][i, :] = data1[:, 2]
        # mags["B"][i, :] = data1[:, 3]
        # mags["V"][i, :] = data1[:, 4]
        # mags["K"][i, :] = data1[:, 5]

        # mags["g"][i, :] = data2[:, 2]
        # mags["r"][i, :] = data2[:, 2] - data2[:, 4]
        # mags["i"][i, :] = data2[:, 2] - data2[:, 5]
        # mags["z"][i, :] = data2[:, 2] - data2[:, 6]

        ages = data1[:, 0] - 9.0  # log yr -> log Gyr, same in all files
        mags["u"][i, :] = data1[:, 1]
        mags["g"][i, :] = data1[:, 2]
        mags["r"][i, :] = data1[:, 3]
        mags["i"][i, :] = data1[:, 4]
        mags["z"][i, :] = data1[:, 5]

        waves, seds, ages_sed = load_binary_ised(filenames_sed[i])
        if wavelengths is None:
            wavelengths = waves
            seds_grid = np.zeros((len(Zvals), nAgeVals, len(waves)), dtype=np.float64)
        seds_grid[i, :, :] = np.float64(seds)

        # # Check if the ages from the .ised file match those from the .1ABmag file (after slicing off the 0-age row)
        assert np.allclose(ages, np.log10(ages_sed) - 9.0, rtol=1e-4, atol=1e-4), (
            f"Error: Age values from .ised file do not match those from .1ABmag file for Z={Zvals[i]}. Please check the files and their formats."
        )

    output_path = (
        pardict["work_dir"]
        + "/"
        + pardict["survey"]
        + "_"
        + pardict["tempname"]
        + "_"
        + "stellar_photometrics.hdf5"
    )

    # write output
    with h5py.File(output_path, "w") as f:
        f["N_Metallicity"] = np.array([len(Zvals)], dtype="int32")
        f["N_LogAgeInGyr"] = np.array([nAgeVals], dtype="int32")
        f["Metallicity_bins"] = np.array(Zvals, dtype="float64")
        f["LogAgeInGyr_bins"] = ages
        f["Wavelengths"] = wavelengths
        f["SEDs"] = seds_grid

        for bandName in bandNames:
            f["Magnitude_" + bandName] = mags[bandName]

        print(f"✅ {output_path} successfully created with full SED integration!")


def main():
    configfile = sys.argv[1]

    # Load configuration parameters
    pardict = ConfigObj(configfile)

    # Generate the stellar_photometrics.hdf5 file using BC03 models
    makeStellarPhotometricsHDF5_BC03(pardict)


if __name__ == "__main__":
    main()
