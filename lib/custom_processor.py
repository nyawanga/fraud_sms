import numpy as np
import pandas as pd
import re

# from sklearn.base import TransformerMixin
# class InputTransformer(TransformerMixin):
class InputTransformer():
    def fit(self, X,y=None, **fit_params ):
        return self

    def clean_text(self, text):
        if re.search(r'(:?.*)(?P<no>07[0-9{8,9}]+)', str(re.sub(r"[^a-zA-Z0-9]+", "", text))):
            has_num = 1
        else:
            has_num = 0

        text = re.sub(r'[^a-zA-Z]', ' ', text) #.replace(r'\s+', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        text_len = len(text)
        try:
            avg_wrd_len = round(text_len / len(text.split()),1)
        except ZeroDivisionError:
            avg_wrd_len = 0.0

        return [text, avg_wrd_len, has_num]

    def transform(self, input_data):
        data = list()
        if len(input_data) == 0:
            return "Error: please enter a valid text, the length of input is too short"
        elif type(input_data) is list:
            for text in input_data:
                row = self.clean_text(text)
                data.append(row)

            data = np.array(data).reshape(len(data),3)
        else:
            data = self.clean_text(input_data)
            data = np.array(data).reshape(1,3)

        df_data = pd.DataFrame(data,
                               columns=["text",'avg_wrd_len','has_num']
                              )
        return df_data

if __name__ == "__main__":
    transformer = InputTransformer()
