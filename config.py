# data.py variables
DATA_FOLDER_PATH = "./data"  # folder containing data file

# plotting variables
PLOT_OUTPUT_PATH = "./plots"  # folder containing plots
SMALL_FONT_SIZE = 14
MEDIUM_FONT_SIZE = 16
LARGE_FONT_SIZE = 18
COLOR1 = "hotpink"  # color of topic 1
COLOR2 = "turquoise"  # color of topic 2

MOD_PLOT_OUTPUT_PATH = "./mod-plots"  # folder containing plots

# modeling variables
LEAST_SQUARES_TRY_TIMES = 50000
FIT_BOUNDS = [(0, 60), (1, 15), (-20, 20),  # mu1, omega1, skew1
              (0, 60), (1, 15), (-20, 20),  # mu2, omega2, skew2
              (0, 60), (1, 15), (-20, 20),  # mu3, omega3, skew3
              (0, 60), (1, 15), (-20, 20),  # mu4, omega4, skew4
              (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # coefficients
FIT_INITIAL_POINT = [40, 13, 0.5,
                     15, 5, 1.0,
                     35, 12, 4,
                     0.2, 4, -6,
                     0.2, 0.1, 0.2, 0.3, 0.1, 0.6]
NDIM = len(FIT_INITIAL_POINT)
LEAST_SQUARES_PERTURB = 0.9
MCMC_PERTURB = 0.1
UPSAMPLE_FACTOR = 10

# parallelizing MCMC variable
N_WORKERS = 8

# parameters regarding the substructure extraction
SUBSTRUCTURES = {
    "jet-shape": {
        "n_bins": 6,
        "req_string_label": "",
        "xlabel": "r",
        "ylabel": r"$\rho(r)$",
        "log": True,
        "title": "Jet Shape",
        "mod_combine_bins": [],
        "xlim_mod": [0, .3],
        "ylim_mod": []
    },
    "jet-frag": {
        "n_bins": 10,
        "req_string_label": "xi",
        "xlabel": r"$\xi$",
        "ylabel": r"$1/N_{jet} dN_{track}/d\xi$",
        "log": True,
        "title": "Jet Fragmentation",
        "mod_combine_bins": [],
        "xlim_mod": [0, 5],
        "ylim_mod": [0.75, 1.5]
    },
    "jet-mass": {
        "n_bins": 50,
        "req_string_label": "",
        "xlabel": "m",
        "ylabel": r"$1/N dN/dm$",
        "log": False,
        "title": "Jet Mass",
        "mod_combine_bins": [[0, 1, 2], [3, 4], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34], [35, 36, 37], [38, 39, 40]],
        "xlim_mod": [0, 30],
        "ylim_mod": [0, 6]
    },
    "jet-splitting": {
        "n_bins": 8,
        "req_string_label": "",
        "xlabel": r"$z_g$",
        "ylabel": r"$1/N dN/dz_{g}$",
        "log": False,
        "title": "Jet Splitting Fraction",
        "mod_combine_bins": [],
        "xlim_mod": [0.1, 0.5],
        "ylim_mod": []
    }
}
