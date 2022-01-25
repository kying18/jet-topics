from multiprocessing import BoundedSemaphore

# data.py variables
DATA_FOLDER_PATH = "./data"  # folder containing data file

# plotting variables
PLOT_OUTPUT_PATH = "./plots"  # folder containing plots
SMALL_FONT_SIZE = 14
MEDIUM_FONT_SIZE = 16
LARGE_FONT_SIZE = 18
COLOR1 = "purple"  # color of topic 1
COLOR2 = "green"  # color of topic 2

# modeling variables
LEAST_SQUARES_TRY_TIMES = 10000
FIT_BOUNDS = [(0, 60), (1, 15), (-20, 20),  # mu1, omega1, skew1
              (0, 60), (1, 15), (-20, 20),  # mu2, omega2, skew2
              (0, 60), (1, 15), (-20, 20),  # mu3, omega3, skew3
              (0, 60), (1, 15), (-20, 20),  # mu4, omega4, skew4
              (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # coefficients
FIT_INITIAL_POINT = [25, 10, 1,
                     10, 10, 4,
                     16, 11, 3,
                     1, 2, 20,
                     0.1, 0.6, 0.2, 0.4, 0.2, 0.02]
NDIM = len(FIT_INITIAL_POINT)
LEAST_SQUARES_PERTURB = 0.5
MCMC_PERTURB = 0.1
UPSAMPLE_FACTOR = 10

# parallelizing MCMC variable
N_WORKERS = 4

# parameters regarding the substructure extraction
SUBSTRUCTURES = {
    "jet-shape": {
        "n_bins": 6,
        "req_string_label": "",
        "xlabel": "r",
        "ylabel": r"$\rho(r)$",
        "log": True,
        "title": "Jet Shape"
    },
    "jet-frag": {
        "n_bins": 10,
        "req_string_label": "xi",
        "xlabel": r"$\xi$",
        "ylabel": r"$1/N_{jet} dN_{track}/d\xi$",
        "log": True,
        "title": "Jet Fragmentation"
    },
    "jet-mass": {
        "n_bins": 50,
        "req_string_label": "",
        "xlabel": "m",
        "ylabel": r"$1/N dN/dm$",
        "log": False,
        "title": "Jet Mass"
    },
    "jet-splitting": {
        "n_bins": 8,
        "req_string_label": "",
        "xlabel": r"$z_G$",
        "ylabel": r"$1/N dN/dz_{G}$",
        "log": False,
        "title": "Jet Splitting Fraction"
    }
}
