import pandas as pd
import numpy as np
from typing import List, Union, Tuple
import os
import re
from scipy.io import loadmat

# Prerequisite information:
joint_names = ["Pelvis", "Hip", "Knee", "Ankle", "FootProgress"]
joint_directions = ["flx/ext", "abd/add", "int/ext rot"]
sides = ["Right", "Left"]


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
    data : np.array
        The data file previously loaded, which is equivalent to the MatLab structure
    structs : List[str]
        All fields in the order to access the desired field.
        For ex, to access 'measurementA' in c.results.side_struct.measurementA,
        structs would look like: ['c', 'results', 'side_struct', 'measurementA']

    Returns
    -------
    np.array
        The data contained in the desired field.
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
def feature_extractor(directory, measurements, output_dir, separated_legs, *joint_names):

    combined_data = []
    data_names = organized(directory)

    for file_number, file in enumerate(data_names, start=0):
        file_path = str()
        file_path = os.path.join(directory, file)  # path of the considered file
        data = load_data(file_path)  # load its data
        side_structs = ["Right", "Left"]

        # -------- Calculate while the info should be extracted separately --------
        if separated_legs == True:
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
    separated_legs: bool,
    output_shape=pd.DataFrame,
    joint_names=["Hip", "Knee", "Ankle", "FootProgress"],
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
        if separated_legs == True:

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


## Helper functions for MinMax_feature_extractor
def collecting_angles(
    all_data: np.ndarray, joint_names: List[str], side: str, min_max: str, separated_legs: bool = True
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Extract angle data and headers for specified joints.

    Parameters
    ----------
    all_data : np.ndarray
        Kind of an array of dictionnary. It is the results when loading MatLab struct into Python.
    joint_names : List[str]
        Names of joints to extract (e.g., ['Hip', 'Knee', 'Ankle']).
    side_struct : str
        Name of the considered side, i.e., 'R' for Right, 'L' for Left
    min_max : str
        Type of considered angles, i.e., 'Min' or 'Max'
    separated_legs : bool, optional
        If True, extract full array; if False, extract scalar values, by default True

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        Joint data arrays and corresponding column headers (e.g., 'Max_Hip_flx/ext')
    """

    joint_data = []
    headers = []

    for joint in joint_names:
        joint_with_side = side + joint

        joint_kin = all_data[0, 0][joint_with_side][0]

        # Extract joint kinematic data
        if separated_legs:
            
            joint_with_side_name = [
                min_max + "_" + joint_with_side[1:] + "_" + direction for direction in joint_directions
            ]
        else:
            # joint_kin = all_data[0, 0][joint_with_side][0][0]
            joint_with_side_name = [min_max + "_" + joint_with_side + "_" + direction for direction in joint_directions]

        joint_data.extend(joint_kin)

        # Create corresponding headers
        headers.extend(joint_with_side_name)

    return joint_data, headers


def collecting_base_sustent(all_data: np.ndarray, side_struct: str):
    """
    Extract base of support (BOS) measure.

    Parameters
    ----------
    all_data : np.ndarray
        MatLab structure loaded in Python.
    side_struct : str
        Side of the body (e.g., 'Right' or 'Left')

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        BOS data and corresponding header
    """

    joint_kin = all_data[0, 0]["maxPreMoyenne"][0]
    joint_kin = np.atleast_1d(joint_kin)
    joint_with_side_name = ["Max_" + side_struct + "_BOS"]

    return joint_kin, joint_with_side_name


def collecting_spatiotemporal_variable(all_data: np.ndarray, measurement: str, side_struct: str = None, separated_legs: bool = True) -> Tuple[np.ndarray, str]:
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
    variable_data = np.atleast_1d(np.asanyarray([all_data[0][0]]))
    
    if separated_legs:
        return variable_data, measurement
    else:
        measurement_with_side = side_struct + measurement
        return variable_data, measurement_with_side


## Main functions for MinMax feature extraction
# if sperataed_legs == True: use this function
def process_measurement(
    all_data: np.ndarray, measurement: str, joint_names: List[str], side_name: str, separated_legs: bool = True
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Process a measurement for separated legs

    Parameters
    ----------
    all_data : np.ndarray
        MatLab structure loaded in Python
    measurement : str
        Name of the measurement to extract (e.g., 'angMinAtFullStance')
    joint_names : List[str]
        List of joint names to consider
    side_name : str
        Side to consider ('Right' or 'Left')
    separated_legs : bool, optional
        If True, extract full arrays; if False, extract scalar values, by default True.

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        Extracted variable data and its corresponding headers
    """
    side = side_name[0]  # 'R' for 'Right', 'L' for 'Left'
    if measurement == "angMinAtFullStance":
        joint_data, headers = collecting_angles(
            all_data, joint_names, side, min_max="Min", separated_legs=separated_legs
        )

    elif measurement == "angMaxAtFullStance":
        joint_data, headers = collecting_angles(
            all_data, joint_names, side, min_max="Max", separated_legs=separated_legs
        )

    elif measurement == "baseSustentation":
        joint_data, headers = collecting_base_sustent(all_data, side)

    else:
        joint_data, header = collecting_spatiotemporal_variable(all_data, measurement, side, separated_legs=separated_legs)
        headers = [header]

    return joint_data, headers


def process_separated_legs(
    data: np.ndarray, measurements: List[str], joint_names: List[str], file_number: int, output_dir: str
) -> List[np.ndarray]:
    """
    Process one data file with legs separated into individual rows.

    Parameters
    ----------
    data : np.ndarray
        Loaded MATLAB data structure
    measurements : List[str]
        Measurements to extract
    joint_names : List[str]
        Joints to process
    file_number : int
        Subject/file number for naming
    output_dir : str
        Directory to save output files

    Returns
    -------
    List[np.ndarray]
        Processed data arrays for both legs
    """

    processed_data = []

    for side_name in sides:
        # Initialize lists to hold combined data and headers
        joint_data_combined = []
        headers_combined = []

        # For each measurement, process and collect data
        for measurement in measurements:
            structs = ["c", "results", side_name, measurement]
            all_data = access_struct(data, structs)

            joint_data, headers = process_measurement(
                all_data, measurement, joint_names, side_name, separated_legs=True
            )

            joint_data_combined.extend(joint_data)
            headers_combined.extend(headers)

        # Flatten and reshape data
        all_scalars = all(np.ndim(x) == 0 for x in joint_data_combined)
        if all_scalars:
            flattened_data = np.array(joint_data_combined, dtype=float)
        else:
            flattened_data = np.concatenate(joint_data_combined)  # Flatten the data
        flattened_data = flattened_data.reshape(1, -1)  # and reshape the joint_data.
        processed_data.append(flattened_data)  # Store flattened data for the considered side.

        # Save individual file
        df = pd.DataFrame(flattened_data, columns=headers_combined)
        filename = f"Subject{file_number + 1}_{side_name[0]}_Lokomat.csv"  # no need to specify pre/post since only the pre files are processed
        df.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"Data extracted for Subject {file_number + 1} (separated legs)")
    return processed_data


def process_combined_legs(
    data: np.ndarray, measurements: List[str], joint_names: List[str], file_number: int, output_dir: str
) -> np.ndarray:
    """
    Process one data file with both legs combined into a single row.

    Parameters
    ----------
    data : np.ndarray
        Loaded MATLAB data structure
    measurements : List[str]
        Measurements to extract
    joint_names : List[str]
        Joints to process
    file_number : int
        Subject/file number for naming
    output_dir : str
        Directory to save output files

    Returns
    -------
    np.ndarray
        Processed data array with both legs
    """
    joint_data_combined = []
    headers_combined = []

    for side_name in sides:
        for measurement in measurements:
            structs = ["c", "results", side_name, measurement]
            all_data = access_struct(data, structs)

            joint_data, headers = process_measurement(
                all_data, measurement, joint_names, side_name, separated_legs=False
            )

            joint_data_combined.extend(joint_data)
            headers_combined.extend(headers)

    # Flatten and reshape data
    all_scalars = all(np.ndim(x) == 0 for x in joint_data_combined)
    if all_scalars:
        flattened_data = np.array(joint_data_combined, dtype=float)
    else:
        flattened_data = np.concatenate(joint_data_combined)  # Flatten the data
    flattened_data = flattened_data.reshape(1, -1)  # and reshape the joint_data.
    # Save individual file
    df = pd.DataFrame(flattened_data, columns=headers_combined)
    filename = (
        f"Subject{file_number + 1}_Lokomat.csv"  # no need to specify pre/post since only the pre files are processed
    )
    df.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"Data extracted for Subject {file_number + 1} (both legs together)")
    return flattened_data


def min_max_feature_extractor(
    directory: str,
    measurements: List[str],
    output_dir: str,
    separated_legs: bool = True,
    joint_names: List[str] = None,
    output_shape: type = pd.DataFrame,
):
    """
    This is the main function that extracts min/max kinematic and spatiotemporal
    features from Lokomat data files.

    This function processes MATLAB structure files containing gait analysis data,
    extracting angle measurements and spatiotemporal parameters for specified joints.

    Parameters
    ----------
    directory : str
        Path to the directory containing the .mat files.
    measurements : List[str]
        List of measurements to extract (e.g., ['pctToeOff', 'baseSustentation']), already defined at the begining of the code file.
        Note: 'angMinAtFullStance' and 'angMaxAtFullStance' are automatically added.
    output_dir : str
        Directory to save the extracted feature CSV files.
    separated_legs : bool, optional
        If True, creates separate rows for each leg.
        If False, combines both legs into a single row.
        By default, True.
    output_shape : type, default=pd.DataFrame
        Return type (pd.DataFrame or np.ndarray) of the combined features for all subjects.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        Combined data for all subjects in specified format.

     Examples
    --------
    >>> min_max_feature_extractor(
    ...     directory='./data/lokomat',
    ...     measurements=['pctToeOff', 'cadence'],
    ...     output_dir='./output',
    ...     separated_legs=True,
    ...     joint_names=['Hip', 'Knee', 'Ankle']
    ... )
    """

    # Set defaults
    if joint_names is None:
        joint_names = joint_names

    # Ensure angle measurements are included
    if "angMaxAtFullStance" not in measurements:
        measurements.insert(0, "angMaxAtFullStance")
    if "angMinAtFullStance" not in measurements:
        measurements.insert(0, "angMinAtFullStance")
    if "angAtFullCycle" in measurements:
        measurements.remove("angAtFullCycle")  # Not relevant for Min/Max extraction

    # Get file list
    data_files = organized(directory)

    # Process all files
    all_processed_data = []

    for file_number, filename in enumerate(data_files):
        file_path = os.path.join(directory, filename)
        data = load_data(file_path)

        if separated_legs:
            processed = process_separated_legs(data, measurements, joint_names, file_number, output_dir)
            all_processed_data.extend(processed)
        else:
            processed = process_combined_legs(data, measurements, joint_names, file_number, output_dir)
            all_processed_data.append(processed)

    # Combine all data
    combined_data = np.vstack(all_processed_data)

    # Get headers from the last processed file (they're consistent across files)
    last_csv = pd.read_csv(
        os.path.join(output_dir, f"Subject{len(data_files)}{'_L' if separated_legs else ''}_Lokomat.csv")
    )
    headers = last_csv.columns.tolist()

    # Save combined file
    df_all = pd.DataFrame(combined_data, columns=headers)
    if separated_legs:
        df_all.to_csv(os.path.join(output_dir, "all_files_separated_legs.csv"), index=False)
    else:
        df_all.to_csv(os.path.join(output_dir, "all_files_combined_legs.csv"), index=False)

    # Return in requested format
    return df_all if output_shape == pd.DataFrame else combined_data


# %% Example of Use

if __name__ == "__main__":
    # Example usage
    result = min_max_feature_extractor(
        directory="/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyData/sample_1/raw_data",
        measurements=["angAtFullCycle", "pctToeOff", "pctToeOffOppose"],
        output_dir="/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyData/sample_1/processed_data",
        separated_legs=False,
        joint_names=["Hip", "Knee", "Ankle"],
    )


# example of use
# mean_feature_extractor(directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Sample',
#                   measurements = ['angAtFullCycle', 'pctToeOff', 'pctToeOffOppose'],
#                   separated_legs = False,
#                   output_dir = 'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Sample',
#                   joint_names = ['Hip', 'Knee', 'Ankle'])
