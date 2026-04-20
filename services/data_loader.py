import pandas as pd

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
        docs.append(doc)

    return docs


df = load_data("data/faturamento_historico.xlsx")
docs = dataframe_to_docs(df)
print(df.head())
