
# -*- coding: utf-8 -*-
import time
import os
os.system('cls' if os.name == 'nt' else 'clear')

import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')

start_time = time.time()

venv_dir = './venv'

#######################################################################################################

#------------------------------------------------------------------------------------------------------

def criar_pastas(diretorio_projeto):
    print(diretorio_projeto)
    print('\n' + 80 * '#')
    print(' Criação de pastas '.center(80,'#'))
    print(80 * '#' + '\n')

    pastas_principais = ['img', 'data', 'doc', 'src', 'model', 'tests','original']

    for pasta in pastas_principais:
        nova_pasta = os.path.join(diretorio_projeto, pasta)
        if not os.path.exists(nova_pasta):
            os.makedirs(nova_pasta)
            print(f"Pasta '{pasta}' criada com sucesso.")
        else:
            print(f"Pasta '{pasta}' já existe.")

#------------------------------------------------------------------------------------------------------

def criar_venv_e_ativar(diretorio_projeto, nome_venv='venv'):

    print('\n' + 80 * '#')
    print(' Criação do ambiente virtual e ativação '.center(80,'#'))
    print(80 * '#' + '\n')


    # Verifica se o diretório do ambiente virtual já existe
    diretorio_venv = os.path.join(diretorio_projeto, nome_venv)
    if not os.path.exists(diretorio_venv):
        # Cria o ambiente virtual
        subprocess.run(["python", "-m", "venv", nome_venv], cwd=diretorio_projeto)
        print("Ambiente virtual criado com sucesso.")

    # Ativa o ambiente virtual
    if os.name == 'nt':  # Para sistemas Windows
        subprocess.run([f"{nome_venv}\\Scripts\\activate"], shell=True, cwd=diretorio_projeto)
    else:  # Para sistemas Unix-like (Linux, macOS)
        subprocess.run([f"source {nome_venv}/bin/activate"], shell=True, cwd=diretorio_projeto)


    print("Ambiente virtual ativado.")

    # Print the current working directory
    os.chdir(venv_dir)
    current_dir = os.getcwd()
    print("Current working directory:", current_dir)

#------------------------------------------------------------------------------------------------------

#######################################################################################################

if __name__ == "__main__":
    inicio = time.time()
    diretorio_projeto = os.getcwd()

    criar_pastas(diretorio_projeto)
    criar_venv_e_ativar(diretorio_projeto)
    

#######################################################################################################
 # end!
#######################################################################################################
print('\n'+' Fim do processamento! '.center(80,'#')+'\n')
end_time = time.time()
execution_time_seconds = end_time - start_time
execution_minutes = int(execution_time_seconds // 60)
execution_seconds = int(execution_time_seconds % 60)
execution_time_str = f"{execution_minutes} minutos, {execution_seconds} segundos"
print("Tempo de execução:", execution_time_str)
