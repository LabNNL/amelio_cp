import os
import pandas as pd
from pandas import DataFrame
import amelio_cp.processing.old_featurExtractor as mat_to_df


class Process:
    def __init__(self):
        pass

    def load_csv(file_path: str) -> DataFrame:
        df = pd.read_csv(file_path, index_col=False)
        return df

    @staticmethod
    def calculate_ROM(df: DataFrame):

        print(df.columns)

        df["ROM_Hip_Sag"] = df["Max_Hip_flx/ext"] - df["Min_Hip_flx/ext"]
        df["ROM_Hip_Frontal"] = df["Max_Hip_abd/add"] - df["Min_Hip_abd/add"]
        df["ROM_Hip_Trns"] = df["Max_Hip_int/ext rot"] - df["Min_Hip_int/ext rot"]
        df["ROM_Knee_Sag"] = df["Max_Knee_flx/ext"] - df["Min_Knee_flx/ext"]
        df["ROM_Ankle_Sag"] = df["Max_Ankle_flx/ext"] - df["Min_Ankle_flx/ext"]

        # TODO: putting the following lines in the function that will do the
        # feature engineering
        df.drop(
            [
                "Min_Knee_abd/add",
                "Min_Knee_int/ext rot",
                "Min_Ankle_abd/add",
                "Min_Ankle_int/ext rot",
                "Max_Knee_abd/add",
                "Max_Knee_int/ext rot",
                "Max_Ankle_abd/add",
                "Max_Ankle_int/ext rot",
            ],
            axis=1,
            inplace=True,
        )

        return df

    @staticmethod
    def load_gps(file_path: str) -> DataFrame:
        df = pd.read_csv(file_path, index_col=False)
        df.drop(columns=["Unnamed: 0", "Post"], inplace=True)
        df.rename(columns={"Pre": "GPS"}, inplace=True)
        return df

    @staticmethod
    def load_demo(file_path: str):

        df = pd.read_excel(file_path)
        df["BMI"] = df["masse"] / ((df["taille"] / 100) ** 2)

        # The numbers are in the order so 1 is walker.
        df.replace(["walker", "cane", "none"], [1, 2, 3], inplace=True)

        # ----- Add Label -----
        df.drop(["delta6MWT", "deltaV"], axis=1, inplace=True)
        # This line excludes some of the features that are in the demographic excel file.
        df.drop(["Patient", "masse", "taille", "sex", "Diagnostique"], axis=1, inplace=True)

        return df

    def load_data(
        self, data_dir: str, output_dir: str, gps_path: str, demographic_path: str, separate_legs: bool = True
    ):
        """
        Load data from raw data

        Parameters
        ----------
        data_path : str
            DESCRIPTION.
        separate_legs : bool, optional
            DESCRIPTION. The default is True.
        gps_path : str
            gait analysis score.

        Returns
        -------
        all_data : TYPE
            DESCRIPTION.

        """
        # file info
        measurements = ["pctSimpleAppuie", "distFoulee", "vitCadencePasParMinute"]
        joint_names = ["Hip", "Knee", "Ankle"]
        side = ["Right", "Left"]

        # Extracting variables from .mat files
        kin_var = mat_to_df.MinMax_feature_extractor(
            directory=data_dir,
            output_dir=output_dir,
            measurements=measurements,
            separate_legs=separate_legs,
            joint_names=joint_names,
        )

        # Calculating ROM from the function above
        kin_var = self.calculate_ROM(kin_var)

        # ----- fixing the values of cadence -----
        kin_var["vitCadencePasParMinute"] *= 2

        # ----- Add GPS to the features -----
        #       Gait Profile Score
        gps = self.load_gps(gps_path)
        all_data = pd.concat((kin_var, gps), axis=1)

        # ----- Add participants demographic variables -----
        demo_var = self.load_demo(demographic_path)

        all_data = pd.concat((all_data, demo_var), axis=1)
        # TODO: concat in one funct

        return all_data

    @staticmethod
    def calculate_MCID(pre_data, post_data, variable, gmfcs_data=None) -> list:
        if variable == "VIT":
            delta_VIT = post_data - pre_data
            MCID_VIT = []
            for i in delta_VIT:
                if i >= 0.1:
                    MCID_VIT.append(1)
                else:
                    MCID_VIT.append(0)
            return pd.Series(MCID_VIT, index=pre_data.index)

        elif variable == "6MWT":
            GMFCS_MCID = {1: range(4, 29), 2: range(4, 29), 3: range(9, 20), 4: range(10, 28)}
            delta_6MWT = post_data - pre_data
            MCID_6MWT = []
            for i in range(len(delta_6MWT)):
                if delta_6MWT.iloc[i] >= max(GMFCS_MCID[gmfcs_data.iloc[i]]):
                    MCID_6MWT.append(1)
                else:
                    MCID_6MWT.append(0)
            return pd.Series(MCID_6MWT, index=gmfcs_data.index)

        else:
            raise ValueError("Variable not recognized. Use 'VIT', or '6MWT'.")

    @staticmethod
    def prepare_dataframe(all_data: DataFrame, condition_to_predict: str, model_name: str):
        """This function prepare the data, to have the features on one side (X)
        and the labels on the other side (y)

        Parameters
        ----------
        all_data : DataFrame
            The whole matrix with features and labels.
        condition_to_predict : str
            Which condition to predict (i.e., 'VIT' or '6MWT' or 'GPS').
        model_name : str
            Model to use (i.e., 'svc' or 'svr').

        Returns
        -------
        tuple
            Features matrix (X) and labels vector (y).

        Raises
        ------
        ValueError
            _description_
        """

        conditions = ["VIT", "6MWT"]  # , "GPS"]

        if condition_to_predict not in conditions:
            raise ValueError("Condition to predict not recognized. Choose either 'VIT', '6MWT', or 'GPS'.")

        conditions.remove(condition_to_predict)
        post_to_remove = [condition + "_POST" for condition in conditions]

        all_data = all_data.drop(columns=post_to_remove)
        all_data = all_data.dropna(axis=0)

        if condition_to_predict == "6MWT":
            gmfcs_data = all_data["GMFCS"]
        else:
            gmfcs_data = None

        if model_name == "svc":
            y = Process.calculate_MCID(
                all_data[condition_to_predict + "_PRE"],
                all_data[condition_to_predict + "_POST"],
                condition_to_predict,
                gmfcs_data,
            )
            X = all_data.drop(columns=[condition_to_predict + "_POST"])
        else:
            y = all_data[condition_to_predict + "_POST"]
            X = all_data.drop(columns=[condition_to_predict + "_POST"])

        return X, y

    @staticmethod
    # TODO: what the best? -> giving paths or dataframes?
    # TODO: splitting in several functions for each condition
    def prepare_data(data_path, condition_to_predict, model_name, features_path=None):
        """function that prepare the data to be suitable for the model (i.e., features, label)

        Parameters
        ----------
        data_path : DataFrame
            Matrix with all the data (i.e., features and labels).
        condition_to_predict : str
            Sets the condition to predict (i.e., 'VIT' or '6MWT' or 'GPS').
        model_name : str
            String to specify the model to use (i.e., 'svc', 'svr').
        features_path : str, optional
            Path to the Excel file containing the features to select and their names (i.e., Max_Knee_flx/ext = Maximum Knee Flexion/Extension), by default None

        Returns
        -------
        X : DataFrame
            Features matrix.
        y : Series
            Labels vector.
        features_names : list
            List of the names of the selected features.

        Raises
        ------
        ValueError
            If the condition to predict is not recognized.
        """

        all_data = Process.load_csv(data_path)

        data, y = Process.prepare_dataframe(all_data, condition_to_predict, model_name)

        # TODO: make a fucntion that takes an array of features to select
        if features_path:
            features = pd.read_excel(features_path)
            selected_features = features["19"].dropna().to_list()
            features_names = features["19_names"].dropna().to_list()
        else:
            selected_features = data.columns.to_list()
            features_names = data.columns.to_list()

        X = data[selected_features]

        return X, y, features_names

    @staticmethod
    def save_df(df, output_path, separate_legs):
        if separate_legs:
            label = "leg_separated"
        else:
            label = "leg_not_separated"

        os.makedirs(output_path, exist_ok=True)
        nb_pp = len(df) // 2
        filename = f"all_data_{nb_pp}pp_{label}.csv"
        filepath = os.path.join(output_path, filename)
        df.to_csv(filepath)
        return filepath

    def _calculate_gsi(data: DataFrame, weight: DataFrame):
        """This function caluclates the Global Strength Index as such:
                    GSI = (1/n)*sum(muscle_torques/weight)

                    with n = number of muscles considered

        Parameters
        ----------
        data : DataFrame
            only the strength values to consider for the calculation of the gsi

        weight : Series
            values of weight to normalised the sum of strength

        Returns
        -------
        DataFrame
            with the GSI for each individual (i.e., row)
        """

        all_gsi = []
        for i in range(len(data)):
            muscle_strenghts = data.iloc[i]
            sum = muscle_strenghts.sum()
            gsi = sum / weight.iloc[i]
            all_gsi.append(gsi)

        all_gsi_df = pd.Series(all_gsi, index=data.index)

        return all_gsi_df

    @staticmethod
    def return_gsi(file_path: str, separate_legs: bool = True):
        """
        Parameters
        ----------
        file_path : str
            Path of the excel file with all the data
            Assuming that:
            Right values should be in the 3-to-7 columns
            Left values should be in the 8-to-13 columns

        separate_legs : bool, optional
            Enables to calculate the gsi for both legs (if False)
            or for each leg (if True),
                        by default True

        Returns
        -------
        DataFrame
            Returns a df with the gsi for each patient (i.e., row)

        Raises
        ------
        TypeError
            if neither True nor False was assigned to separate_legs
        """

        data = pd.read_excel(file_path)
        weight = data["weight"]

        if separate_legs == True:
            right_gsi = Process._calculate_gsi(data.iloc[:, 2:7], weight)
            left_gsi = Process._calculate_gsi(data.iloc[:, 7:], weight)

            return right_gsi, left_gsi

        elif separate_legs == False:
            gsi = Process._calculate_gsi(data.iloc[:, 2:], weight)
            return gsi

        else:
            raise TypeError("'separate_legs' was not correctly set, it should be eithe True or False.")
