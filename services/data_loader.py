import pandas as pd
import json

def load_data(path):
    df = pd.read_excel(path)

    # clean currency
    def clean_currency(value):
        return float(str(value).replace(".", "").replace(",", "."))

    df["Vlr Total"] = df["Vlr Total"].apply(clean_currency)

    return df


def dataframe_to_docs(df):
    docs = []

    for _, row in df.iterrows():
        doc = f"""
        Establishment: {row['Estabelecimento']}.
        Total Value: {row['Vlr Total']}.
        Quantity: {row['Qde']}.
        Month: {row['Anomes']}.
        Category: {row['depara']}.
        """
        docs.append(json.dumps(doc))

    return docs


