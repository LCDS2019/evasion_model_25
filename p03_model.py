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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def limpar_valores_br(df, colunas):
    for col in colunas:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def classificar_risco(prob):
    if prob < 0.4:
        return 'Baixo risco'
    elif prob < 0.7:
        return 'Médio risco'
    else:
        return 'Alto risco'


base_2023 = pd.read_csv("base_2023.csv", sep=';', decimal = ',')
base_2024 = pd.read_csv("base_2024.csv", sep=';', decimal = ',')

num_vars = ['idade', 'frequencia', 'media_notas', 'n_reprovacoes',
            'renda_familiar', 'tem_bolsa', 'trabalha', 'apoio_familiar']
cat_vars = ['genero', 'curso']

df_treino = base_2023[base_2023['evadiu'].notna()].copy()
df_treino = limpar_valores_br(df_treino, num_vars)

X_train = df_treino.drop(columns=['evadiu', 'aluno_id', 'ano', 'situacao_estudante'])
y_train = df_treino['evadiu']

preprocessador = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars)
], remainder='passthrough')

ratio = (y_train == 0).sum() / (y_train == 1).sum()

modelo = Pipeline([
    ('preprocessamento', preprocessador),
    ('xgb', XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=ratio
    ))
])

modelo.fit(X_train, y_train)

y_proba = modelo.predict_proba(X_train)[:, 1]
limiar = 0.55
y_pred = (y_proba >= limiar).astype(int)

y_proba = modelo.predict_proba(X_train)[:, 1]

print('\n' + ' Avaliação do Modelo '.center(80, '#') + '\n')
print(f"Acurácia:       {accuracy_score(y_train, y_pred):.2f}")
print(f"Precisão:       {precision_score(y_train, y_pred):.2f}")
print(f"Recall:         {recall_score(y_train, y_pred):.2f}")
print(f"F1 Score:       {f1_score(y_train, y_pred):.2f}")
print(f"AUC ROC:        {roc_auc_score(y_train, y_proba):.2f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_train, y_pred))


limiares = [0.2, 0.3, 0.4, 0.5, 0.6]
print("\nAnálise de thresholds:")
for limiar in limiares:
    y_pred = (y_proba >= limiar).astype(int)
    print(f"\nLimiar: {limiar:.2f}")
    print(f"Precisão: {precision_score(y_train, y_pred):.2f}")
    print(f"Recall:   {recall_score(y_train, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_train, y_pred):.2f}")

#------------------------------------------------------------------------------------------------------

limiares = np.arange(0.1, 0.91, 0.05)
precisoes = []
recalls = []
f1_scores = []

for limiar in limiares:
    y_pred = (y_proba >= limiar).astype(int)
    precisoes.append(precision_score(y_train, y_pred))
    recalls.append(recall_score(y_train, y_pred))
    f1_scores.append(f1_score(y_train, y_pred))


plt.figure(figsize=(10, 6))
plt.plot(limiares, precisoes, label='Precisão', marker='o', color='#1f77b4')
plt.plot(limiares, recalls, label='Recall', marker='s', color='#ff7f0e')
plt.plot(limiares, f1_scores, label='F1 Score', marker='^', color='#2ca02c')

plt.axvline(0.58, color='red', linestyle='--', label='Limiar escolhido (0.58)')

plt.xlabel('Limiar de Decisão')
plt.ylabel('Métrica')
plt.title('Precisão, Recall e F1 Score por Limiar de Decisão')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('img/model.png', dpi=300, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------------------------------

df_pred = base_2024.copy()
df_pred = limpar_valores_br(df_pred, num_vars)

X_pred = df_pred.drop(columns=['evadiu', 'aluno_id', 'ano', 'situacao_estudante'])
df_pred['prob_evasao'] = modelo.predict_proba(X_pred)[:, 1]

data_service_evasao = df_pred[['aluno_id', 'curso', 'situacao_estudante', 'frequencia', 'media_notas', 'renda_familiar', 'prob_evasao']]
#print(data_service_evasao.sort_values(by='prob_evasao', ascending=False))

data_service_evasao['risco_evasao'] = data_service_evasao['prob_evasao'].apply(classificar_risco)

print(data_service_evasao['risco_evasao'].value_counts())

data_service_evasao.to_csv('base_results_XGboost.csv',sep='|', index=False, decimal = ',')

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
