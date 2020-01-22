import iris
import iris.coord_categorisation
import numpy as np
import matplotlib.pyplot as plt
import math

# This script contains a set of functions which helps to create a weigthed mean
# based upon multiple metrics
# It is based around netCDF files using iris.

###############################################
# Data Processing
###############################################


def get_mdls_reals(cube):

    mdl = cube.attributes['model_id']
    real = 'r{}i{}p{}'.format(cube.attributes['realization'],
                              cube.attributes['initialization_method'],
                              cube.attributes['physics_version'])

    return mdl + real


def remove_mdlreal(cubes_list, mdls):
    """"
    This removes a model from the list
    """
    for i in range(len(cubes_list)):
        for cube in cubes_list:
            if get_mdls_reals(cube) in mdls:
                cubes_list.remove(cube)

    return


def constrain_mdl_obs_time(mdls, obs):
    """
    This constrains all the models to the temporal range of the observations
    """

    years = obs.coord('year').points
    for cube in mdls:
        years = np.intersect1d(years, cube.coord('year').points)
    con = iris.Constraint(year=years)

    out_mdls = []
    for cube in mdls:
        con_cube = cube.extract(constraint=con)
        out_mdls.append(con_cube)

    out_obs = obs.extract(constraint=con)

    return out_mdls, out_obs


def constrain_mdl_time(mdls):
    """
    This constrains the models so they all have the same temporal range
    """

    years = mdls[0].coord('year').points
    for cube in mdls[1:]:
        years = np.intersect1d(years, cube.coord('year').points)
    con = iris.Constraint(year=years)

    out_mdls = []
    for cube in mdls:
        con_cube = cube.extract(constraint=con)
        out_mdls.append(con_cube)

    return out_mdls


def normalise_mdls_obs(mdls, obs):
    """
    Performs a standard normalisation of models to the mean and
    std of the obs (z score) for metrics which are non scalar
    """
    mean = np.mean(obs.data)
    std = np.std(obs.data)

    mdls_norm = []
    for cube in mdls:
        mdls_norm.append(cube.copy((cube.data - mean) / std))

    obs_norm = obs.copy((obs.data - mean) / std)

    return mdls_norm, obs_norm


def normalise_mdls_obs_single_value(mdls, obs):
    """
    Performs a standard normalisation of models to the mean and
    std of the obs (and models) (z score) for metrics which are
    scalar
    """
    values = list(mdls.values())
    values.append(obs)
    mean = np.mean(values)
    std = np.std(values)

    mdls_norm = mdls.copy()
    for mdl in mdls.keys():
        mdls_norm[mdl] = (mdls[mdl] - mean) / std

    obs_norm = (obs - mean) / std

    return mdls_norm, obs_norm


def sort_cubes(cubes, print_mdls=0):
    """
    # This ensures that the list of models is alphabetical
    """
    mdls = [get_mdls_reals(cube) for cube in cubes]
    sorted_cubes = [cube for a, cube in sorted(zip(mdls, cubes))]
    mdls.sort()
    if print_mdls == 0:
        return sorted_cubes
    if print_mdls == 1:
        return sorted_cubes, mdls


def normalise_MM_mdls_obs(mdls, obs):
    
    min_ = np.min(obs.data)
    max_ = np.max(obs.data)
    for m in mdls:
        if np.min(m.data) < min_:
            min_ = np.min(m.data)
        if np.max(m.data) > max_:
            max_ = np.max(m.data)
        
    
    mdls_norm = []
    for cube in mdls:
        mdls_norm.append(cube.copy((1 * (cube.data - min_) / (max_ - min_)) - 0))
        
    obs_norm = obs.copy((1 * (obs.data - min_) / (max_ - min_)) - 0)
    
    return mdls_norm, obs_norm


def normalise_MM_mdls_obs_single_value(mdls, obs):
    
    values = list(mdls.values())
    values.append(obs)
    o_max = np.nanmax(values)
    o_min = np.nanmin(values)
    
    mdls_norm = mdls.copy()
    for mdl in mdls.keys():
        mdls_norm[mdl] = (1 * (mdls[mdl] - o_min) / (o_max - o_min)) - 0
        
    obs_norm = (1 * (obs - o_min) / (o_max - o_min)) - 0
    
    return mdls_norm, obs_norm


def cube_has_coord(cube, coord_id):
    # Returns boolean on whether the cube has a coord of specified coord_name

    # Check if coord_id is either a name of coord or coord instance
    if isinstance(coord_id, str):
        id_type = 0
    elif isinstance(coord_id, iris.coords.Coord):
        id_type = 1
    else:
        raise ValueError(
            'Specified input for coordinate is neither a coord name or a coord instance')
    if not isinstance(cube, iris.cube.Cube):
        raise ValueError('First argument must be a cube')

    # Get coord names
    coords = cube.coords()
    coord_names = []
    for coord in coords:
        coord_names.append(coord.name())

    if id_type == 0:    # Coord name
        if coord_id in coord_names:
            output = True
        else:
            output = False
    elif id_type == 1:    # Coord type
        if coord_id in coords:
            output = True
        else:
            output = False

    return output


def constrain_year(cubes, t0, t1):
    # Constrains the cube year to be between t0 and t1 inclusive

    if not isinstance(cubes, list):
        cubes = [cubes]

    for cube in cubes:
        if not cube_has_coord(cube, 'year'):
            iris.coord_categorisation.add_year(cube, 'time')

    if len(cubes) == 1:
        cube = cubes[0]
        idx0 = np.where(cube.coord('year').points == t0)[0][0]
        idx1 = np.where(cube.coord('year').points == t1)[0][0]
        output = cube[idx0:idx1 + 1]
    else:
        output = []
        for cube in cubes:
            idx0 = np.where(cube.coord('year').points == t0)[0][0]
            idx1 = np.where(cube.coord('year').points == t1)[0][0]
            output.append(cube[idx0:idx1 + 1])

    return output


###############################################
# Weighting Functions
###############################################


def independence(mdls, sigma=0.1, data_type_cube=True):
    """
    This calculates the independence of the models for a given metric
    where the metric is NOT single valued, e.g. the time series of CO2 
    concentration.
    ------Input------
    mdls (list) : The models (either as iris cubes or as numpy arrays)
    sigma (float) : The value of sigma_s
    data_type_cube (boolean) : Whether the data in models are in iris cubes
    -----Returns-----
    S (np.array 2D) : The inter model similarity
    W (np.array 1D) : The weight per model from the similarity calculation
    """

    sigma_s = sigma

    # Can first calculate inter model distances S and D
    S = np.zeros((len(mdls), len(mdls)))

    # Weightings W dims=num_models
    W = np.zeros((len(mdls), 1))

    for i, model_i in enumerate(mdls):
        if not data_type_cube:
            i_data = model_i
        else:
            i_data = model_i.data
        for j, model_j in enumerate(mdls):
            if i != j:
                if not data_type_cube:
                    j_data = model_j
                else:
                    j_data = model_j.data
                s = math.exp(-((i_data - j_data) ** 2).sum() / (len(j_data) * sigma_s ** 2))
                S[i, j] = s

    for ii in range(len(mdls)):
        w = 1 / (1 + np.nansum(S[ii], 0))
        W[ii] = w

    W /= np.nansum(W)

    return S, W


def independence_single_value(values, sigma=0.70):
    """
    This calculates the independence of the models for a given metric
    where the metric is single valued, e.g. the slope of a gradient.
    ------Input------
    values (list) : The single values for each model.
    sigma (float) : The value of sigma_s
    -----Returns-----
    S (np.array 2D) : The inter model similarity
    W (np.array 1D) : The weight per model from the similarity calculation
    """

    sigma_s = sigma

    # Can first calculate inter model distances S and D
    S = np.zeros((len(values), len(values)))

    # Weightings W dims=num_models
    W = np.zeros((len(values), 1))

    for i, model_i in enumerate(values):
        i_data = model_i

        for j, model_j in enumerate(values):
            if i != j:
                j_data = model_j
                s = math.exp(-((i_data - j_data) ** 2).sum() / (1 * sigma_s ** 2))
                S[i, j] = s

    for ii in range(len(values)):
        w = 1 / (1 + np.nansum(S[ii], 0))
        W[ii] = w

    W /= np.nansum(W)

    return S, W


def performance(mdls, obs, sigma=0.1, data_type_cube=True):
    """
    This calculates the performace of the models for a given metric
    where the metric is NOT single valued, e.g. the time series of CO2
    concentration.
    ------Input------
    mdls (list) : The models (either as iris cubes or as numpy arrays)
    obs (iris.cube.Cube, or np.array) : The observation to compare the 
                                        models against
    sigma (float) : The value of sigma_sd
    data_type_cube (boolean) : Whether the data in models are in iris cubes
    -----Returns-----
    W (np.array 1D) : The weight per model from the performance calculation
    """

    sigma_d = sigma

    D = np.zeros((len(mdls), 1))

    if not data_type_cube:
        obs_data = obs
    else:
        obs_data = obs.data

    for i, model_i in enumerate(mdls):
        if not data_type_cube:
            i_data = model_i
        else:
            i_data = model_i.data
        d = math.exp(- ((i_data - obs_data) ** 2).sum() / (len(i_data) * sigma_d ** 2))
        D[i] = d

    W = D / np.nansum(D)

    return W


def performance_single_value(values, obs, sigma=0.36):
    """
    This calculates the performance of the models for a given metric
    where the metric is single valued, e.g. the slope of a gradient.
    ------Input------
    values (list) : The single values for each model.
    obs (float) : The observation to compare the models against
    sigma (float) : The value of sigma_s
    -----Returns-----
    W (np.array 1D) : The weight per model from the performance calculation
    """
    sigma_d = sigma

    D = np.zeros((len(values), 1))

    W = np.zeros((len(values), 1))

    for i, model_i in enumerate(values):
        i_data = model_i
        d = math.exp(- ((i_data - obs) ** 2).sum() / (sigma_d ** 2))
        D[i] = d

    W = D / np.nansum(D)

    return W

###############################################
# Plotting Functions
###############################################


def plot_s(S, mdls):
    """
    This plots the similarities between models
    ------Input------
    S (np.array 2D) : The inter model similarity
    mdls (list) : Names of the models (str)
    """

    f, ax = plt.subplots()
    plt.pcolor(1 - S)
    cax = plt.colorbar()
    plt.xticks([i + 0.5 for i in list(range(len(mdls)))], mdls, rotation='vertical')
    plt.yticks([i + 0.5 for i in list(range(len(mdls)))], mdls)

    f.show()

    return


def plot_d(D, mdls):
    """
    This plots the performance of models
    ------Input------
    D (np.array 2D) : The model's performance
    mdls (list) : Names of the models (str)
    """
    plt.figure()
    plt.bar(mdls, D.reshape(-1))

    plt.xticks(rotation='vertical')
    plt.ylabel('Weighting')
    plt.show()
    return


def plot_cubes(cubes):
    
    plt.figure(figsize=(8,6))
    for cube in cubes:
        iris.quickplot.plot(cube.coord('year'), cube, label=get_mdls_reals(cube))
    plt.legend()
    plt.show()

###############################################
# Projection Functions
###############################################

###############################################
# Testing Functions
###############################################

