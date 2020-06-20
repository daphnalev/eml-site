import random
import threading
from typing import List
import sys
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import pandas as pd
import os
import scipy.linalg as la
import numpy as np
import time
import warnings
from sklearn import preprocessing
import logging
from enum import Enum


####
def one_line_warning(message, category, filename, lineno, file=None, line=None):
    return '%s: line %s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = one_line_warning
black = False
white = True


####
class Parser:
    def __init__(self):
        # todo private in the future -  add "__" before all
        self.x_names = np.array([])
        self.y_names = np.array([])
        self.dummies_names = np.array([])
        self.dummies_dict = dict()
        self.dummies = pd.DataFrame()
        self.data = None
        self.dropped_rows = set()
        self.empty_cell_indicators = set({None, ""})
        self.label_encoder = None
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

    def load(self, file_name, ftype="csv"):
        self.__init__()
        if ftype == "csv":
            self.data = pd.read_csv(file_name)
        if ftype == "xlsx":
            self.data = pd.read_excel(file_name)
        # self.x_names = np.array(self.data.columns[:-1])
        self.x_names = np.array(self.data.columns)
        # self.y_names = np.array(self.data.columns[-1])

    def get_all_names(self):
        return self.data.columns

    def parse(self, d):
        print("Starting preprocessing")
        t = time.time()
        X,X_bool, Y, x_names, dummies_name, removed_rows, must_cols = self.setter(d)
        preprocessingFuncs = [self.basic_preprocessing4_new]  # , self.basic_preprocessing2]
        results = []
        # todo threading
        for func in preprocessingFuncs:
            results.append(func(X,X_bool, Y, x_names, dummies_name, removed_rows, must_cols))
        i, s = 0, 0
        for j, res in enumerate(results):
            if res[0].size > s:
                s = res[0].size
                i = j
        f0 = self.num_of_features()
        self.X, self.Y, self.x_names, self.dummies_names, self.label_encoder = results[i]
        f1 = self.num_of_features()
        print("Finished preprocessing within {t}[sec] with:".format(t=time.time() - t))
        print("{f1}/{f0} features remains\n{co1}/{co0} rows remains".format(f0=f0, f1=f1, co0=self.data.shape[0],
                                                                            co1=self.X.shape[0]))

    def set_dummies(self, features: List[str]):

        dummies_arr = np.array(features)
        existence_data = np.in1d(dummies_arr, self.data.columns)
        if False in existence_data:
            warnings.warn(f"Set aborted\nNo such features: {dummies_arr[np.where(existence_data==False)[0]]}")
            return
        self.dummies_names = dummies_arr
        temp_x = self.x_names[~np.in1d(self.x_names, dummies_arr)]  # remove dummies from X
        temp_y = self.y_names[~np.in1d(self.y_names, dummies_arr)]  # remove dummies from Y
        if temp_x != self.x_names:
            warnings.warn(f"X changed from {self.x_names} to {temp_x}")
            self.x_names = temp_x
        if temp_y != self.y_names:
            warnings.warn(f"Y changed from {self.__y_names} to {temp_y}")
            self.y_names = temp_y

    def set_features(self, features: List[str]):
        features_arr = np.array(features)
        existence_data = np.in1d(features_arr, self.data.columns)
        if False in existence_data:
            warnings.warn(f"Set aborted\nNo such features: {features_arr[np.where(existence_data==False)[0]]}")
            return
        self.x_names = features_arr
        temp_dummies = self.dummies_names[~np.in1d(self.dummies_names, features_arr)]  # remove X from dummies from
        temp_y = self.y_names[~np.in1d(self.y_names, features_arr)]  # remove X from y
        if temp_dummies != self.dummies_names:
            warnings.warn(f"Dummies changed from {self.__dummies_names} to {temp_dummies}")
            self.dummies_names = temp_dummies
        if temp_y != self.y_names:
            warnings.warn(f"Y changed from {self.__y_names} to {temp_y}")
            self.y_names = temp_y

    def __create_dummies(self, rows, dummies_names):
        if len(dummies_names) and len(rows):
            return pd.get_dummies(self.data.iloc[rows][dummies_names])
        return pd.DataFrame()
    def __create_dummies_new(self,X, rows, dummies_names):
        if len(dummies_names) and len(rows):
            return pd.get_dummies(X.iloc[rows][dummies_names])
        return pd.DataFrame()

    def set_y(self, names: List[str]):
        y_arr = np.array(names)
        existence_data = np.in1d(y_arr, self.data.columns)
        if False in existence_data:
            warnings.warn(f"Set aborted\nNo such features: {y_arr[np.where(existence_data==False)[0]]}")
            return
        self.y_names = y_arr
        temp_dummies = self.dummies_names[~np.in1d(self.dummies_names, y_arr)]  # remove Y from dummies
        temp_x = self.x_names[~np.in1d(self.x_names, y_arr)]  # remove Y from X
        if temp_dummies != self.dummies_names:
            warnings.warn(f"Dummies changed from {self.__dummies_names} to {temp_dummies}")
            self.dummies_names = temp_dummies
        if temp_x != self.x_names:
            warnings.warn(f"X changed from {self.y_names} to {temp_x}")
            self.x_names = temp_x

    def get_y_names(self):
        return self.y_names

    def get_features_names(self):
        return np.append(self.x_names, self.dummies_names)

    def num_of_features(self):
        return len(self.x_names) + len(self.dummies_names)

    def getXY(self):
        if self.X is None or self.Y is None:
            warnings.warn(f"Warning! - seems like you didn't parse")
        return self.X, self.Y

    def correlation(self, display=False, eps=.98):
        """
        Check features correlation greater then given threshold.
        1. One should check if the returned unhandled_features are corrupted.
        2. One should consider to work only on the remains features (or replace features f with feature f'
           such that f' is in  correlation_dict[f]
        :param display: to display correlation(regardless eps)
        :param eps: threshold
        :return: remains - set of features to work on,
                 dropped - features that has correlated feature that is in remains
                 correlation_dict - mapping between feature that's in remains to all the features correlated to each
                                    other that where dropped
                 unhandled_features - features that there was a problem check for correlation
        """

        df = (self.data[self.get_features_names()])
        values_transpose = df.values.T
        unhandled_features = []
        new_values = []
        for index, col in enumerate(values_transpose):
            changed = np.array([])
            try:
                empty_indices = np.where(np.in1d(col, self.empty_cell_indicators) == True)[0]
                s = len(empty_indices)
                if s != 0:
                    changed = np.random.choice(1e5, s)  # todo random around mean?
                    col[empty_indices] = changed
                col = col.astype(np.float64)
                new_values.append(col)
            except:
                try:
                    label_encoder = preprocessing.LabelEncoder()
                    label_encoder.fit(col)
                    new_col = label_encoder.transform(col)
                    new_col[empty_indices] = changed
                    new_values.append(new_col)
                except:
                    unhandled_features.append(index)
        correlation = np.corrcoef(np.array(new_values))
        if display:
            f = plt.figure(figsize=(19, 15))
            plt.matshow(correlation, fignum=f.number)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Correlation Matrix', fontsize=16)
            plt.show()
        highly_correlated = np.where(correlation > eps)
        indices = np.where(highly_correlated[0] < highly_correlated[1])[
            0]  # consider only half of the symmetric matrix
        correlation_dict = dict()
        for index in indices:
            key = highly_correlated[0][index]
            val = highly_correlated[1][index]
            if key in correlation_dict.keys():
                correlation_dict[key].append(val)
            else:
                correlation_dict[key] = [val]
        self.__combine_keys(correlation_dict)
        remains = set()
        dropped = set()
        correlation_names_dict = dict()
        for k, v in correlation_dict.items():
            key = df.columns[k]
            values = df.columns[v].values
            remains.add(key)
            dropped.update(set(values))
            correlation_names_dict[key] = values
        return np.array(list(remains)), np.array(list(dropped)), correlation_names_dict, df.columns[
            np.array(unhandled_features)].values

    def __delete_key(self, d, key):
        """given a key check if it's in some lower key value if so return the first key
        else return -1"""
        nxt = -1
        for k in sorted(d.keys(), reverse=True):
            if k >= key:
                continue
            if key in d[k]:
                nxt = k
                break
        if nxt != -1:
            try:
                to_check = [key] + d[key]
                if False in np.in1d(d[nxt], to_check):
                    return
                else:
                    del d[key]
                    self.__delete_key(d, nxt)
            except:
                # already del kk
                return
        return

    def __combine_keys(self, d):
        """
        Combine keys if transitive where d[key] is a list of ints.
        if D[a]=[b,c] and D[b]=[c] => delete b but if D[a]=[b,c,d],D[b]=[c,e] don't delete
        where D===d
        :param d: dictionary
        :return: None. Adjust given dictionary
        """
        for kk in sorted(d.keys(), reverse=True):
            self.__delete_key(d, kk)

    def remove_rows(self, rows):
        self.dropped_rows.update(rows)

    def add_empty_cell_indicators(self, indicators: List[str]):
        self.empty_cell_indicators = self.empty_cell_indicators | set(indicators)

    def remove_empty_cell_indicators(self, indicators: List[str]):
        self.empty_cell_indicators = self.empty_cell_indicators - set(indicators)

    def get_empty_cell_indicators(self):
        return self.empty_cell_indicators

    def __get_bad_rows(self, df, black_or_white):
        where = np.where(df == black_or_white)
        if not where[0].any():
            return np.array([])
        diff = where[0][1:] - where[0][:-1]
        indices = np.where(diff > 0)[0]
        black_rows = [set()] * df.shape[0]  # np.empty(B_X.shape[0],dtype=object)
        last = 0
        for index in indices:
            row = where[0][last]
            cols = where[1][last:index + 1]
            black_rows[row] = set(cols)  # cols.tolist()
            last = index + 1
        # add the last one
        row = where[0][last]
        cols = where[1][last:]
        black_rows[row] = set(cols)
        return black_rows

    def basic_preprocessing2(self):
        # todo can improve run time - sort rows by num of black
        # todo then delete N first rows where min(BlackCol)=N then iterate to see if delete remain blacks
        # todo by row/col each iteration
        t = time.time()
        X = (self.data[self.get_features_names()])
        indicators = np.array(list(self.empty_cell_indicators))
        bool_x_vals = ~np.isin(X.values, indicators)
        black_rows = self.__get_bad_rows(bool_x_vals, black)
        white_rows = self.__get_bad_rows(bool_x_vals, white)
        black_cols = self.__get_bad_rows(bool_x_vals.T, black)
        white_cols = self.__get_bad_rows(bool_x_vals.T, white)
        removed_rows = self.dropped_rows.copy()
        removed_cols = []
        bad_col = len(black_rows) + 1
        bad_row = len(black_cols) + 1
        rows_length = [0] * len(black_rows)
        cols_length = [0] * len(black_cols)
        empty = [set()] * bool_x_vals.shape[1]
        if np.array(black_cols).any():
            while black_cols != empty:
                for i in range(len(black_rows)):
                    if len(black_rows[i]) > 0:
                        rows_length[i] = len(white_rows[i])
                    else:
                        rows_length[i] = bad_row
                for i in range(len(black_cols)):
                    if len(black_cols[i]) > 0:
                        cols_length[i] = len(white_cols[i])
                    else:
                        cols_length[i] = bad_col
                min_row_val = min(rows_length)
                min_col_val = min(cols_length)
                if min_col_val <= min_row_val:
                    drop_col = cols_length.index(min_col_val)
                    removed_cols.append(drop_col)
                    white_cols[drop_col] = set()
                    black_cols[drop_col] = set()
                    for i in range(len(white_rows)):
                        try:
                            black_rows[i].remove(drop_col)
                        except KeyError:
                            pass
                        try:
                            white_rows[i].remove(drop_col)
                        except KeyError:
                            pass
                else:
                    drop_row = rows_length.index(min_row_val)
                    removed_rows.add(drop_row)
                    white_rows[drop_row] = set()
                    black_rows[drop_row] = set()
                    for i in range(len(white_cols)):
                        try:
                            black_cols[i].remove(drop_row)
                        except KeyError:
                            pass
                        try:
                            white_cols[i].remove(drop_row)
                        except KeyError:
                            pass
        # all_rows = np.arange(X.shape[0])
        # rows = np.delete(all_rows, np.array(removed_rows))
        all_columns = np.arange(X.shape[1])
        cols = np.delete(all_columns, removed_cols)
        # x, y =
        # print(f"Preprocessing Time: {time.time() - t} seconds")
        return self.__pre_helper(list(removed_rows), cols)
        # cols_names = X.columns[cols].values
        # existence_dummies = np.in1d(cols_names, self.dummies_names)
        # x_cols_names = cols_names[~existence_dummies]
        # dummies_cols = cols_names[existence_dummies]
        # self.dummies_names = dummies_cols
        # convert = []
        # new_x = X.iloc[rows]
        # for name in x_cols_names:
        #     try:
        #         new_x[[name]].astype(np.float64)
        #     except:
        #         convert.append(name)
        # convert = np.array(convert)
        # existence_convert = np.in1d(x_cols_names, convert)
        # self.x_names = x_cols_names[~existence_convert]
        # self.dummies_names = np.append(self.dummies_names, convert)
        # dummies = self.__create_dummies(rows)
        # new_x = new_x[self.x_names]
        # new_y = self.data[self.y_names].iloc[rows]
        # try:
        #     new_y.astype(np.float64)
        # except:
        #     self.label_encoder = preprocessing.LabelEncoder()
        #     self.label_encoder.fit(new_y.values)
        #     new_y = pd.DataFrame(self.label_encoder.transform(new_y), columns=self.y_names)
        # if not dummies.empty:
        #     new_x.join(dummies)
        # if removed_rows != [] or removed_cols != []:
        #     warnings.warn(f"\nRemoved rows: {removed_rows}\n and columns:{removed_cols}")
        # print(f"Preprocessing Time: {time.time() - t} seconds")
        # # new_x.join(new_y).to_csv('example_data/after.csv')
        # return new_x, new_y

    def basic_preprocessing4(self):
        t = time.time()
        X = (self.data[self.get_features_names()]).copy()
        indicators = np.array(list(self.empty_cell_indicators))
        bool_x_vals = ~np.isin(X.values, indicators)
        black_cols = self.__get_bad_rows(bool_x_vals.T, black)
        white_cols = self.__get_bad_rows(bool_x_vals.T, white)
        removed_rows = self.dropped_rows.copy()
        chosen_cols = set()
        iters = range(len(black_cols))
        if len(iters) == 0:
            chosen_cols = set(np.arange(len(self.get_features_names())))
        else:
            for _ in iters:
                best_col = -1
                best_value = 0
                for col_index in range(len(black_cols)):
                    if col_index in chosen_cols:
                        continue
                    added_value = len(white_cols[col_index]) - (
                        len(chosen_cols) * len(black_cols[col_index] - removed_rows))
                    if added_value > best_value:
                        best_value = added_value
                        best_col = col_index
                if best_value > 0:
                    chosen_cols.add(best_col)
                    removed_rows.update(black_cols[best_col])
                    continue
                break
        return self.__pre_helper(list(removed_rows), list(chosen_cols))

    def basic_preprocessing4_new(self, X,X_bool, Y, x_names, dummies_name, removed_rows, must_cols):
        t = time.time()

        # indicators = np.array(list(self.empty_cell_indicators))
        # X_bool = ~np.isin(X.values, indicators)
        print("NEW")
        black_cols = self.__get_bad_rows(X_bool.T, black)
        print("BLACK")
        white_cols = self.__get_bad_rows(X_bool.T, white)
        print("White")
        all_removed_rows = self.dropped_rows.copy()
        all_removed_rows.update(removed_rows)
        chosen_cols = must_cols
        iters = range(len(black_cols))
        if len(iters) == 0:
            chosen_cols = set(np.arange(len(self.get_features_names())))
        else:
            for _ in iters:
                best_col = -1
                best_value = 0
                for col_index in range(len(black_cols)):
                    if col_index in chosen_cols:
                        continue
                    added_value = len(white_cols[col_index]) - (
                        len(chosen_cols) * len(black_cols[col_index] - removed_rows))
                    if added_value > best_value:
                        best_value = added_value
                        best_col = col_index
                if best_value > 0:
                    chosen_cols.add(best_col)
                    removed_rows.update(black_cols[best_col])
                    continue
                break
        return self.__pre_helper_new(list(removed_rows), list(chosen_cols),X, dummies_name,Y)

    def __pre_helper_new(self,bad_rows, good_cols_lst,X, dummies_names,Y):
        # X = (self.data[self.get_features_names()])
        all_rows = np.arange(X.shape[0])
        rows = np.delete(all_rows, np.array(bad_rows))
        cols_names = X.columns[good_cols_lst].values
        existence_dummies = np.in1d(cols_names, dummies_names)
        x_cols_names = cols_names[~existence_dummies]
        dummies_cols = cols_names[existence_dummies]
        dummies_names = dummies_cols
        convert = []
        new_x = X.iloc[rows].copy()
        for name in x_cols_names:
            try:
                new_x[[name]].astype(np.float64)
            except:
                convert.append(name)
        convert = np.array(convert)
        existence_convert = np.in1d(x_cols_names, convert)
        x_names = x_cols_names[~existence_convert]
        dummies_names = np.append(dummies_names, convert)
        dummies = self.__create_dummies_new(X,rows, dummies_names)
        new_x = new_x[x_names]
        new_y = Y.iloc[rows].copy()
        label_encoder = None
        try:
            new_y.astype(np.float64)
        except:
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(new_y.values)
            new_y = pd.DataFrame(label_encoder.transform(new_y), columns=Y.columns)
        if not dummies.empty:
            new_x.join(dummies)
        # print(f"Preprocessing Time: {time.time() - t} seconds")
        return new_x, new_y, x_names, dummies_names, label_encoder

    def __pre_helper(self, bad_rows, good_cols_lst):
        X = (self.data[self.get_features_names()])
        all_rows = np.arange(X.shape[0])
        rows = np.delete(all_rows, np.array(bad_rows))
        cols_names = X.columns[good_cols_lst].values
        existence_dummies = np.in1d(cols_names, self.dummies_names)
        x_cols_names = cols_names[~existence_dummies]
        dummies_cols = cols_names[existence_dummies]
        dummies_names = dummies_cols
        convert = []
        new_x = X.iloc[rows].copy()
        for name in x_cols_names:
            try:
                new_x[[name]].astype(np.float64)
            except:
                convert.append(name)
        convert = np.array(convert)
        existence_convert = np.in1d(x_cols_names, convert)
        x_names = x_cols_names[~existence_convert]
        dummies_names = np.append(dummies_names, convert)
        dummies = self.__create_dummies(rows, dummies_names)
        new_x = new_x[x_names]
        new_y = self.data[self.y_names].iloc[rows].copy()
        label_encoder = None
        try:
            new_y.astype(np.float64)
        except:
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(new_y.values)
            new_y = pd.DataFrame(label_encoder.transform(new_y), columns=self.y_names)
        if not dummies.empty:
            new_x.join(dummies)
        # print(f"Preprocessing Time: {time.time() - t} seconds")
        return new_x, new_y, x_names, dummies_names, label_encoder



        # def remove_rows_by_values(self, values: List[str]):
        #     rows = set()
        #     for val in values:
        #         rows.update(np.where(self.__data.values == val)[0])
        #     self.__dropped_rows.update(rows)
        # def check_features(self, features: List[str]):
        #     result = dict()
        #     data_features = self.get_features_names()
        #     for feature in features:
        #         if feature in data_features:
        #             result[feature] = True
        #         else:
        #             result[feature] = False
        #     return result
        # def basic_preprocessing3(self):
        #     t = time.time()
        #     X = (self.data[self.get_features_names()])
        #     indicators = np.array(list(self.empty_cell_indicators))
        #     bool_x_vals = ~np.isin(X.values, indicators)
        #     black_rows = self.__get_bad_rows(bool_x_vals, black)
        #     white_rows = self.__get_bad_rows(bool_x_vals, white)
        #     black_cols = self.__get_bad_rows(bool_x_vals.T, black)
        #     white_cols = self.__get_bad_rows(bool_x_vals.T, white)
        #     removed_rows = []
        #     removed_cols = []
        #     bad_col = len(black_rows) + 1
        #     bad_row = len(black_cols) + 1
        #     rows_length = [0] * len(black_rows)
        #     cols_length = [0] * len(black_cols)
        #     empty = [set()] * bool_x_vals.shape[1]
        #     while black_cols != empty:
        #         for i in range(len(black_rows)):
        #             rows_length[i] = len(black_rows[i]) - len(white_rows[i])
        #         for i in range(len(black_cols)):
        #             cols_length[i] = len(black_cols[i]) - len(white_cols[i])
        #
        #         max_row_val = max(rows_length)
        #         max_col_val = max(cols_length)
        #         if max_col_val >= max_row_val:
        #             drop_col = cols_length.index(max_col_val)
        #             removed_cols.append(drop_col)
        #             white_cols[drop_col] = set()
        #             black_cols[drop_col] = set()
        #             for i in range(len(white_rows)):
        #                 try:
        #                     black_rows[i].remove(drop_col)
        #                 except KeyError:
        #                     pass
        #                 try:
        #                     white_rows[i].remove(drop_col)
        #                 except KeyError:
        #                     pass
        #         else:
        #             drop_row = rows_length.index(max_row_val)
        #             removed_rows.append(drop_row)
        #             white_rows[drop_row] = set()
        #             black_rows[drop_row] = set()
        #             for i in range(len(white_cols)):
        #                 try:
        #                     black_cols[i].remove(drop_row)
        #                 except KeyError:
        #                     pass
        #                 try:
        #                     white_cols[i].remove(drop_row)
        #                 except KeyError:
        #                     pass
        #     all_columns = np.arange(X.shape[1])
        #     cols = np.delete(all_columns, removed_cols)
        #     return self.__pre_helper(removed_rows, cols)
        #     # all_rows = np.arange(X.shape[0])
        #     # rows = np.delete(all_rows, np.array(removed_rows))
        #     #
        #     # cols = np.delete(all_columns, removed_cols)
        #     # cols_names = X.columns[cols].values
        #     # existence_dummies = np.in1d(cols_names, self.dummies_names)
        #     # x_cols_names = cols_names[~existence_dummies]
        #     # dummies_cols = cols_names[existence_dummies]
        #     # self.dummies_names = dummies_cols
        #     # convert = []
        #     # new_x = X.iloc[rows]
        #     # for name in x_cols_names:
        #     #     try:
        #     #         new_x[[name]].astype(np.float64)
        #     #     except:
        #     #         convert.append(name)
        #     # convert = np.array(convert)
        #     # existence_convert = np.in1d(x_cols_names, convert)
        #     # self.x_names = x_cols_names[~existence_convert]
        #     # self.dummies_names = np.append(self.dummies_names, convert)
        #     # dummies = self.__create_dummies(rows)
        #     # new_x = new_x[self.x_names]
        #     # new_y = self.data[self.y_names].iloc[rows]
        #     # try:
        #     #     new_y.astype(np.float64)
        #     # except:
        #     #     self.label_encoder = preprocessing.LabelEncoder()
        #     #     self.label_encoder.fit(new_y.values)
        #     #     new_y = pd.DataFrame(self.label_encoder.transform(new_y), columns=self.y_names)
        #     # if not dummies.empty:
        #     #     new_x.join(dummies)
        #     # if removed_rows != [] or removed_cols != []:
        #     #     warnings.warn(f"\nRemoved rows: {removed_rows}\n and columns:{removed_cols}")
        #     # print(f"Preprocessing Time: {time.time() - t} seconds")
        #     # # new_x.join(new_y).to_csv('example_data/after.csv')
        #     # return new_x, new_y

        # todo private in the future
        # def basic_preprocessing(self):
        #     t = time.time()
        #     X = (self.data[self.x_names].join(self.dummies)).values
        #     indicators = np.array(list(self.empty_cell_indicators))
        #     B_X = ~np.isin(X, indicators)
        #     removed_rows = []
        #     removed_cols = []
        #     while False in B_X:
        #         cols = np.sum(B_X, axis=0)
        #         rows = np.sum(B_X, axis=1)
        #         vote_cols = np.zeros(cols.shape)
        #         vote_rows = np.zeros(rows.shape)
        #         for x in range(len(cols)):
        #             for y in range(len(rows)):
        #                 if cols[x] >= rows[y]:
        #                     vote_cols[x] += 1
        #                 else:
        #                     vote_rows[y] += 1
        #         c_max = np.max(vote_cols)
        #         r_max = np.max(vote_rows)
        #         if c_max > r_max:
        #             to_remove = np.where(vote_cols == c_max)[0][0]
        #             adjusted_to_remove = np.sum(removed_cols <= to_remove) + to_remove
        #             removed_cols.append(adjusted_to_remove)
        #             B_X = np.delete(B_X, to_remove, 1)
        #         else:
        #             to_remove = np.where(vote_rows == r_max)[0][0]
        #             adjusted_to_remove = np.sum(removed_rows <= to_remove) + to_remove
        #             removed_rows.append(adjusted_to_remove)
        #             B_X = np.delete(B_X, to_remove, 1)
        #     print(f"First: {time.time() - t}")
        #     return removed_rows, removed_cols

        # todo private in the future

    def stats(self):
        if self.X is None:
            return
        return self.X.describe().transpose()

    def setter(self, dictionary):
        """
        dictionary of feature_name:[type,missingHandling,isMust]
        type - values: [feature/categorical feature/output]
        handling - how to handle missing data. values: [delete, median, mean,value]
        missingTypes  - what data is consider missing ("",None,Nan etc)
        isMust - indicator if to include the feature
        :param dictionary:
        :return:
        """
        Y = pd.DataFrame()
        removed_rows = set()
        x_names = []
        dummies_name = []
        must_cols = []
        X = self.data.copy()
        X_bool = pd.DataFrame(np.ones(X.shape, dtype=bool), columns=X.columns)
        for feature in dictionary:
            col_type, handling, missing_types, inclusion = dictionary[feature]
            if inclusion == Parser.ColInclusion.drop:
                X.drop(feature, axis=1, inplace=True)
                X_bool.drop(feature,axis=1,inplace=True)
                continue
            col = X[feature]
            rows = np.where(np.isin(col, missing_types))[0]
            if len(rows):
                # means there are missing rows
                if handling == Parser.ColHandling.delete:
                    X_bool[feature] = False
                    if inclusion == Parser.ColInclusion.must:
                        # pass
                        removed_rows.add(rows)
                elif handling == Parser.ColHandling.median:
                    try:
                        col[rows] = np.median(col)
                    except:
                        warnings.warn(f"Invalid input for median in feature:{f},Change handling to delete ")
                        X_bool[feature] = False
                        if inclusion == Parser.ColInclusion.must:
                            # pass
                            removed_rows.add(rows)

                elif handling == Parser.ColHandling.mean:
                    try:
                        col[rows] = np.mean(col)
                    except:
                        warnings.warn(f"Invalid input for mean in feature:{f},Change handling to delete ")
                        X_bool[feature] = False
                        if inclusion == Parser.ColInclusion.must:
                            # pass
                            removed_rows.add(rows)

                else:
                    # is some value
                    col[rows] = handling
            if col_type == Parser.ColType.output:
                Y = pd.DataFrame(col, columns=[feature])
                X.drop(feature, axis=1, inplace=True)
                X_bool.drop(feature, axis=1, inplace=True)
            else:
                # X = X.append(pd.DataFrame(col, columns=[feature]))
                if col_type == Parser.ColType.feature:
                    x_names.append(feature)
                else:
                    dummies_name.append(feature)
            if inclusion == Parser.ColInclusion.must and col_type != Parser.ColType.output:
                must_cols.append(feature)

        return X, X_bool, Y, x_names, dummies_name, removed_rows, must_cols

    class ColType(Enum):
        feature = 1
        categorical_feature = 2
        output = 3

    class ColHandling(Enum):
        delete = 1
        median = 2
        mean = 3

    class ColInclusion(Enum):
        must = 1
        possible = 2
        drop = 3
