import time
import os
os.system('cls' if os.name == 'nt' else 'clear')
start_time = time.time()

#######################################################################################################
# imports
#######################################################################################################

import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

base_2023 = pd.read_csv("base_2023.csv",sep=';', decimal=',')

print(base_2023.columns.tolist())

def limpar_valores_br(df, colunas):
    for col in colunas:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

base_2023 = pd.read_csv("base_2023.csv", sep=';')

num_vars = ['idade', 'frequencia', 'media_notas', 'n_reprovacoes',
            'renda_familiar', 'tem_bolsa', 'trabalha', 'apoio_familiar']
cat_vars = ['genero', 'curso'] 

df = base_2023[
    (base_2023['situacao_estudante'] == 'Ingressante') &
    (base_2023['evadiu'].notna())
].copy()

df = limpar_valores_br(df, num_vars)

X = df.drop(columns=['evadiu', 'aluno_id', 'ano', 'situacao_estudante'])
y = df['evadiu']

correlacoes = df[num_vars + ['evadiu']].corr()['evadiu'].drop('evadiu').sort_values(ascending=False)

preprocessador = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars)
], remainder='passthrough')

modelo = Pipeline([
    ('preprocessamento', preprocessador),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

modelo.fit(X, y)

ohe_features = modelo.named_steps['preprocessamento'].get_feature_names_out()
num_features = [col for col in X.columns if col not in cat_vars]
feature_names = modelo.named_steps['preprocessamento'].get_feature_names_out()

importancias = modelo.named_steps['xgb'].feature_importances_
importancia_df = pd.DataFrame({'variavel': feature_names, 'importancia': importancias})
importancia_df = importancia_df.sort_values(by='importancia', ascending=False)

print("\n Correlação com evasão:")
print(correlacoes)

print("\n Importância das variáveis segundo XGBoost:")
print(importancia_df.head(15))


#######################################################################################################
# end!
#######################################################################################################
print(' Fim do processamento! '.center(80,'#')+'\n')
end_time = time.time()
execution_time_seconds = end_time - start_time
execution_minutes = int(execution_time_seconds // 60)
execution_seconds = int(execution_time_seconds % 60)
execution_time_str = f"{execution_minutes} minutos, {execution_seconds} segundos"
print("Tempo de execução:", execution_time_str)
