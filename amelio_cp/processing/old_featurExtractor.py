import pandas as pd
import numpy as np
import os
import re
from scipy.io import loadmat


# READ .mat  FILES IN PYTHON
def load_data(file):

    data = loadmat(file)
    print("--------------------------------")
    print("The data is now loaded!")

    return data


# Organize mat files in the order of PreLokomat and then PostLokomat.
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)]


def organized(directory):

    pre_files = [
        f for f in os.listdir(directory) if f.endswith("eLokomat.mat")
    ]  # Collects all the PreLokomat files and stores them in the pre_files list.
    post_files = [
        f for f in os.listdir(directory) if f.endswith("stLokomat.mat")
    ]  # Collects all the PostLokomat files and stores them in the post_files list.
    mat_files_pre_sorted = sorted(
        pre_files, key=natural_sort_key
    )  # Arg from list will be split (i.e., ['PreLokomat', 2, '.mat']) and then will sort files according to their id (here,2).
    mat_files_post_sorted = sorted(post_files, key=natural_sort_key)  # IDEM for Post_ files
    mat_files_sorted = (
        mat_files_pre_sorted + mat_files_post_sorted
    )  # Regrouping both pre and post list to have the whole dataset
    print("The files are sorted in the following order: pre training, then post training!")

    return mat_files_sorted


# Access to the selected feature, set at the last items of structs
def access_struct(data, structs):
    """
    Parameters
    ----------
    data : array
        the data file previousluy loaded, which is equivalent to the MatLab structure
    structs :
        all fields in the order to access to the wished field
        for ex, is want to access 'measurementA' in c.results.side_struct.measurementA, structs would look like: ['c', 'results', 'side_struct', 'measurementA']
    """
    for struct in structs:
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            data = data[0, 0][struct]
        else:
            data = data[struct]
    return data


# Extract all features
# Example:
# measurements = ['angAtFullCycle', 'pctToeOff', 'pctToeOffOppose']
# joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']

"""
def feature_extractor (directory, measurements, output_dir, *joint_names):
    
    combined_data = []
    data_names = organized(directory)
    for file_number, file in enumerate(data_names, start = 0):
        file_path = str()
        file_path = os.path.join(directory, file)
        data = load_data(file_path)

        side_structs = ['Right', 'Left']
        joint_data = []

        for side_struct in side_structs:

            for measurement in measurements:
                
                structs = ['c', 'results', side_struct, measurement]
                all_data = access_struct(data,structs)

                if 'angAtFullCycle' in measurements:

                    joint_data = np.empty((100,0))
                    if measurement == 'angAtFullCycle':

                        sides = side_struct[0]
                        for joint in (joint_names):
                            for side in sides:
                                joint_with_side = side + joint
                                joint_kin = all_data[0,0][joint_with_side][0][0]
                                joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                                joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

                    else: 
                        variable = all_data[0][0]
                        filler = np.full((99,1), np.nan)
                        variable = np.vstack((variable, filler))
                        joint_data = np.concatenate((joint_data,variable), axis = 1)
                
                else:
                    variable = all_data[0][0]
                    #joint_data = np.concatenate((joint_data,variable), axis = 1)
                    joint_data.append(variable)

        print("The data for the Subject %d is extracted." %(file_number+1))
        combined_data.append(joint_data)
        joint_data = pd.DataFrame(joint_data).T
        joint_data.to_csv(output_dir + '\Subject%d_Lokomat.csv' % (file_number +1), header = False, index = False)
        print("The data is successfully saved!")
    all_files = pd.DataFrame(combined_data)
    all_files.to_csv(output_dir + r'all_files.csv', header = False, index = False)
    return combined_data


# This function output the specified measurements for each side the diffrence
# with the function <<feature_extractor>> is that this function calculates the output separately.
"""


# TODO: decide whether two following func are needed
def feature_extractor(directory, measurements, output_dir, separate_legs, *joint_names):

    combined_data = []
    data_names = organized(directory)

    for file_number, file in enumerate(data_names, start=0):
        file_path = str()
        file_path = os.path.join(directory, file)  # path of the considered file
        data = load_data(file_path)  # load its data
        side_structs = ["Right", "Left"]

        # -------- Calculate while the info should be extracted separately --------
        if separate_legs == True:
            for side_struct in side_structs:
                joint_data = []
                for measurement in measurements:

                    structs = [
                        "c",
                        "results",
                        side_struct,
                        measurement,
                    ]  # list of leveled fields, in MatLab would be: c.results.side_struct.measurement
                    all_data = access_struct(
                        data, structs
                    )  # accessing to the data in the order predetermined by structs
                    # at this point, all_data is a big matrix with all values of the variable 'measurement', in the three directions for every joints

                    if "angAtFullCycle" in measurements:

                        joint_data = np.empty((100, 0))  # initiates an array
                        if measurement == "angAtFullCycle":

                            side = side_struct[0]
                            for joint in joint_names:  # for each joint of the list, do
                                joint_with_side = side + joint  # creates a name, ex: 'LHip'
                                joint_kin = all_data[0, 0][joint_with_side][0][
                                    0
                                ]  # goes and finds the appropriate value for LHip
                                joint_kin = np.reshape(joint_kin, (100, 3), order="F")
                                joint_data = np.concatenate((joint_data, joint_kin), axis=1)

                        else:
                            variable = all_data[0][0]
                            filler = np.full((99, 1), np.nan)
                            variable = np.vstack((variable, filler))
                            joint_data = np.concatenate((joint_data, variable), axis=1)

                    else:
                        variable = all_data[0][0]
                        joint_data.append(variable)

                combined_data.append(joint_data)
                joint_data_side = pd.DataFrame(joint_data).T
                joint_data_side.to_csv(
                    output_dir + "Subject%d_%s_Lokomat.csv" % ((file_number + 1), side_struct[0]),
                    header=False,
                    index=False,
                )

            print("The data for the Subject %d is extracted, separated legs." % (file_number + 1))

        # -------- Calculate while the info should be extracted together --------
        else:
            joint_data = []
            for side_struct in side_structs:
                for measurement in measurements:

                    structs = ["c", "results", side_struct, measurement]
                    all_data = access_struct(data, structs)

                    if "angAtFullCycle" in measurements:

                        joint_data = np.empty((100, 0))
                        if measurement == "angAtFullCycle":

                            sides = side_struct[0]
                            for joint in joint_names:
                                for side in sides:
                                    joint_with_side = side + joint
                                    joint_kin = all_data[0, 0][joint_with_side][0][0]
                                    joint_kin = np.reshape(joint_kin, (100, 3), order="F")
                                    joint_data = np.concatenate((joint_data, joint_kin), axis=1)

                        else:
                            variable = all_data[0][0]
                            filler = np.full((99, 1), np.nan)
                            variable = np.vstack((variable, filler))
                            joint_data = np.concatenate((joint_data, variable), axis=1)

                    else:
                        variable = all_data[0][0]
                        # joint_data = np.concatenate((joint_data,variable), axis = 1)
                        joint_data.append(variable)

            combined_data.append(joint_data)
            joint_data_side = pd.DataFrame(joint_data).T
            joint_data_side.to_csv(output_dir + "Subject%d_Lokomat.csv" % (file_number + 1), header=False, index=False)
            print("The data for the Subject %d is extracted, both legs together." % (file_number + 1))

    all_files = pd.DataFrame(combined_data)
    all_files.to_csv(output_dir + r"all_files.csv", header=False, index=False)
    return combined_data


#  Mean calculation is not correct in this funcion since it computes mean for the whole cycle while it should be during stance phase.
def mean_feature_extractor(
    *,
    directory,
    measurements,
    output_dir,
    separate_legs: bool,
    output_shape=pd.DataFrame,
    joint_names=["Hip", "Knee", "Ankle", "FootProgress"]
):
    # The function calculates the mean value of kinematic parameters during the whole gait cycle.
    # Outout: mean value of each
    count = 0
    combined_data = []
    data_names = organized(directory)

    for file_number, file in enumerate(data_names, start=0):
        file_path = str()
        file_path = os.path.join(directory, file)
        data = load_data(file_path)
        side_structs = ["Right", "Left"]

        # -------- Calculate while the info should be extracted separately --------
        if separate_legs == True:

            for side_struct in side_structs:
                joint_data = []
                header = []
                count += 1

                for measurement in measurements:
                    structs = ["c", "results", side_struct, measurement]
                    all_data = access_struct(data, structs)

                    if "angAtFullCycle" in measurements:
                        if measurement == "angAtFullCycle":

                            sides = side_struct[0]
                            for joint in joint_names:
                                for side in sides:
                                    joint_with_side = side + joint
                                    joint_kin = all_data[0, 0][joint_with_side][0][0]
                                    joint_kin = np.reshape(joint_kin, (100, 3), order="F")
                                    joint_kin = np.mean(joint_kin, axis=0)
                                    joint_data.append(joint_kin)
                                    joint_with_side = [
                                        joint_with_side[1:] + "_" + direction
                                        for direction in ["flx/ext", "abd/add", "int/ext rot"]
                                    ]
                                    header.extend(joint_with_side)

                        else:
                            header.append(measurement)
                            variable = np.asanyarray([all_data[0][0]])
                            joint_data.append(variable)

                    else:
                        header.append(measurement)
                        variable = np.asanyarray([all_data[0][0]])
                        joint_data.append(variable)

                joint_data = np.concatenate(joint_data)  # Flatten the joint_data to get it ready for reshapeing.
                joint_data = joint_data.reshape(1, -1)
                combined_data.append(joint_data)
                joint_data_side = pd.DataFrame(joint_data, columns=header)
                joint_data_side.to_csv(
                    output_dir + "Subject%d_%s_Lokomat.csv" % ((file_number + 1), side_struct[0]), index=False
                )

            print("The data for the Subject %d is extracted, separated legs." % (file_number + 1))

        # -------- Calculate while the info should be extracted together --------
        else:
            joint_data = []
            header = []
            count += 1
            for side_struct in side_structs:

                for measurement in measurements:

                    structs = ["c", "results", side_struct, measurement]
                    all_data = access_struct(data, structs)

                    if "angAtFullCycle" in measurements:

                        if measurement == "angAtFullCycle":

                            sides = side_struct[0]
                            for joint in joint_names:
                                for side in sides:
                                    joint_with_side = side + joint
                                    joint_kin = all_data[0, 0][joint_with_side][0][0]
                                    joint_kin = np.reshape(joint_kin, (100, 3), order="F")
                                    joint_kin = np.mean(joint_kin, axis=0)
                                    joint_data.append(joint_kin)
                                    joint_with_side = [
                                        joint_with_side + "_" + direction
                                        for direction in ["flx/ext", "abd/add", "int/ext rot"]
                                    ]
                                    header.extend(joint_with_side)

                        else:
                            header.append(measurement)
                            variable = np.asanyarray([all_data[0][0]])
                            joint_data.append(variable)

                    else:
                        header.append(measurement)
                        variable = np.asanyarray([all_data[0][0]])
                        joint_data.append(variable)

            joint_data = np.concatenate(joint_data)  # Flatten the joint_data to get it ready for reshapeing.
            joint_data = joint_data.reshape(1, -1)
            combined_data.append(joint_data)
            joint_data_side = pd.DataFrame(joint_data, columns=header)
            joint_data_side.to_csv(output_dir + "Subject%d_Lokomat.csv" % (file_number + 1), index=False)
            print("The data for the Subject %d is extracted, both legs together." % (file_number + 1))

    combined_data = np.concatenate(combined_data)  # Flatten the joint_data to get it ready for reshapeing.
    combined_data = combined_data.reshape(count, -1)
    all_files = pd.DataFrame(combined_data, columns=header)
    all_files.to_csv(output_dir + r"all_files.csv", index=False)

    if output_shape == pd.DataFrame:
        return all_files  # while outputting pandas dataframe
    else:
        return combined_data  # While outputting numpy array


# %% Calculates the minimum and maximum values of joints degrees of freedom.
# TODO: find a way to handle both 'if' states
def collecte_angles(all_data, joint_names, side_struct, min_max: str, separate_legs: bool):
    """
    Extract angle data and headers for all joints and, if separate_legs=True, for one side (Right/Left)

    Parameters
    ----------
    all_data : array
        kind of an array of dictionnary. It is the results when loading MatLab struct into Python.
    joint_names : list
        list of all joints considered
    side_struct : string
        name of the side considered, i.e., 'Right' or 'Left'
    min_max : str
        type of angles considered, i.e., 'Min' or 'Max'
    separate_legs : bool
        to indicate if calculation should be done for each leg (i.e., True), or for both legs (i.e., False)

    Returns
    -------
    returns a df with all the joint data and their names (i.e., headers, e.g., 'Max_Hip_flx/ext')
    """

    side = side_struct[0]  # can be defined before calling the function if separate_legs=True
    joint_data = []
    headers = []

    for joint in joint_names:
        joint_with_side = side + joint
        # Extract joint kinematic data
        joint_kin = all_data[0, 0][joint_with_side][0]
        joint_data.append(joint_kin)

        # Create corresponding headers
        joint_with_side_name = [
            min_max + "_" + joint_with_side[1:] + "_" + direction for direction in ["flx/ext", "abd/add", "int/ext rot"]
        ]
        headers.extend(joint_with_side_name)

    return joint_data, headers


def collecting_base_sustent(all_data, side_struct):

    joint_kin = all_data[0, 0]["maxPreMoyenne"][0]
    joint_with_side_name = ["Max_" + side_struct + "_" + "BOS"]

    return joint_kin, joint_with_side_name


def collect_spatiotemporal_variable(all_data, measurement):
    """
    Extract spatiotemporal variable (e.g., cadence)

    Parameters
    ----------
    all_data : np.ndarray
        MatLab structure loaded in Python
    measurement : str
        Name of the measurement to extract (e.g., 'cadence')

    Returns
    -------
    Tuple[np.ndarray, str]
        Extracted variable data and its
    """
    variable_data = np.asanyarray([all_data[0][0]])
    return variable_data, measurement


def process_measurement_separated_legs(all_data, measurement, joint_names, side_name):
    """
    Process a measurement for separated legs

    Parameters
    ----------
    all_data : np.ndarray
        MatLab structure loaded in Python
    measurement : str
        Name of the measurement to extract (e.g., 'angMinAtFullStance')
    joint_names : list
        List of joint names to consider
    side_name : str
        Side to consider ('Right' or 'Left')

    Returns
    -------
    Tuple[np.ndarray, list]
        Extracted variable data and its corresponding headers
    """
    if measurement == "angMinAtFullStance":
        joint_data, headers = collecte_angles(all_data, joint_names, side_name, min_max="Min")
    elif measurement == "angMaxAtFullStance":
        joint_data, headers = collecte_angles(all_data, joint_names, side_name, min_max="Max")
    elif measurement == "baseSustentation":
        joint_data, headers = collecting_base_sustent(all_data, side_name)
    else:
        joint_data, header = collect_spatiotemporal_variable(all_data, measurement)
        headers = [header]

    return joint_data, headers


def MinMax_feature_extractor(
    *,
    directory,
    measurements,
    output_dir,
    separate_legs: bool,
    output_shape=pd.DataFrame,
    joint_names=["Pelvis", "Hip", "Knee", "Ankle", "FootProgress"]
):
    # Outout: min and max value of each
    count = 0
    combined_data = []
    side_structs = ["Right", "Left"]
    data_names = organized(directory)
    if "angMaxAtFullStance" not in measurements:
        measurements.insert(0, "angMaxAtFullStance")
    if "angMinAtFullStance" not in measurements:
        measurements.insert(0, "angMinAtFullStance")

    for file_number, file in enumerate(data_names, start=0):
        file_path = str()
        file_path = os.path.join(directory, file)
        data = load_data(file_path)

        # -------- Calculate while the info should be extracted separately --------
        if separate_legs == True:

            for side_struct in side_structs:
                joint_data_glob = []
                headers_glob = []
                count += 1

                for measurement in measurements:
                    structs = ["c", "results", side_struct, measurement]
                    all_data = access_struct(data, structs)
                    headers_glob

                    if measurement == "angMinAtFullStance":
                        joint_data, headers = collecting_angles(all_data, joint_names, side_struct, min_max="Min")
                        joint_data_glob.extend(joint_data)
                        headers_glob.extend(headers)

                    elif measurement == "angMaxAtFullStance":
                        joint_data, headers = collecting_angles(all_data, joint_names, side_struct, min_max="Max")
                        joint_data_glob.extend(joint_data)
                        headers_glob.extend(headers)

                    elif measurement == "baseSustentation":
                        joint_kin, header = collecting_base_sustent(all_data, side_struct)
                        joint_data_glob.append(joint_kin)
                        headers_glob.extend(header)

                    else:
                        headers_glob.append(measurement)
                        spatiotemporal_variable = np.asanyarray([all_data[0][0]])
                        joint_data_glob.append(spatiotemporal_variable)

                joint_data_glob = np.concatenate(
                    joint_data_glob
                )  # Flatten the joint_data to get it ready for reshapeing.
                joint_data_glob = joint_data_glob.reshape(1, -1)
                combined_data.append(joint_data_glob)
                joint_data_side = pd.DataFrame(joint_data_glob, columns=headers_glob)
                joint_data_side.to_csv(
                    output_dir + "Subject%d_%s_Lokomat.csv" % ((file_number + 1), side_struct[0]), index=False
                )

            print("The data for the Subject %d is extracted, separated legs." % (file_number + 1))

        # -------- Calculate while the info should be extracted together --------
        else:
            joint_data_glob = []
            headers_glob = []
            count += 1
            for side_struct in side_structs:

                for measurement in measurements:

                    structs = ["c", "results", side_struct, measurement]
                    all_data = access_struct(data, structs)

                    if measurement == "angMinAtFullStance":
                        sides = side_struct[0]
                        for joint in joint_names:
                            for side in sides:
                                joint_with_side = side + joint
                                joint_kin = all_data[0, 0][joint_with_side][0][0]
                                joint_data_glob.append(joint_kin)
                                joint_with_side = [
                                    "Min" + "_" + joint_with_side + "_" + direction
                                    for direction in ["flx/ext", "abd/add", "int/ext rot"]
                                ]
                                headers_glob.extend(joint_with_side)

                    elif measurement == "angMaxAtFullStance":
                        sides = side_struct[0]
                        for joint in joint_names:
                            for side in sides:
                                joint_with_side = side + joint
                                joint_kin = all_data[0, 0][joint_with_side][0][0]
                                joint_data_glob.append(joint_kin)
                                joint_with_side = [
                                    "Max" + "_" + joint_with_side + "_" + direction
                                    for direction in ["flx/ext", "abd/add", "int/ext rot"]
                                ]
                                headers_glob.extend(joint_with_side)

                    elif measurement == "baseSustentation":
                        joint_kin, header = collecting_base_sustent(all_data, side_struct)
                        joint_data_glob.append(joint_kin)
                        headers_glob.extend(header)

                    else:
                        headers.append(measurement)
                        spatiotemporal_variable = np.asanyarray([all_data[0][0]])
                        joint_data_glob.append(spatiotemporal_variable)

            joint_data_glob = np.concatenate(joint_data_glob)  # Flatten the joint_data to get it ready for reshapeing.
            joint_data_glob = joint_data_glob.reshape(1, -1)
            combined_data.append(joint_data_glob)
            joint_data_side = pd.DataFrame(joint_data_glob, columns=headers_glob)
            joint_data_side.to_csv(output_dir + "Subject%d_Lokomat.csv" % (file_number + 1), index=False)
            print("The data for the Subject %d is extracted, both legs together." % (file_number + 1))

    combined_data = np.concatenate(combined_data)  # Flatten the joint_data to get it ready for reshapeing.
    combined_data = combined_data.reshape(count, -1)
    all_files = pd.DataFrame(combined_data, columns=headers_glob)
    all_files.to_csv(output_dir + r"all_files.csv", index=False)

    if output_shape == pd.DataFrame:
        return all_files  # while outputting pandas dataframe
    else:
        return combined_data  # While outputting numpy array


# example of use
# mean_feature_extractor(directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Sample',
#                   measurements = ['angAtFullCycle', 'pctToeOff', 'pctToeOffOppose'],
#                   separate_legs = False,
#                   output_dir = 'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Sample',
#                   joint_names = ['Hip', 'Knee', 'Ankle'])
